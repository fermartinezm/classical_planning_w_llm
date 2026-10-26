from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math # For infinity

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
    # Ensure fact has at least as many parts as args, unless args has wildcards at the end
    if len(parts) < len(args) and '*' not in args[len(parts):]:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It sums the estimated costs for each unachieved goal predicate
    independently, taking the minimum cost over all capable rovers. The cost
    for achieving a goal predicate (communicating data) involves two main phases:
    1. Obtaining the data (sampling soil/rock or taking an image).
    2. Communicating the data from a location visible to the lander.
    Travel costs between waypoints are estimated using precomputed shortest paths
    for each rover.

    # Assumptions
    - Action costs are uniform (cost 1).
    - The heuristic ignores the state of the rover's store (empty/full) for sampling costs.
    - The heuristic simplifies the image calibration process: it assumes calibration
      is needed before taking an image unless a suitable camera on the rover is
      already calibrated in the current state. Calibration status is lost after
      taking an image.
    - The heuristic assumes that if a sample or image is needed, the required
      `at_soil_sample`, `at_rock_sample`, `visible_from` facts are present in
      the initial state or static facts, making the goal achievable in principle.
      If a required waypoint (sample location, imaging location, calibration location,
      communication location) is unreachable by a rover, that rover is excluded
      from the minimum cost calculation for that goal. If no rover can reach
      the necessary locations for a goal, that goal is considered unreachable
      by this heuristic, and a large cost is assigned.

    # Heuristic Initialization
    - Parses static facts to identify:
        - All rovers, waypoints, landers, stores, cameras, modes, objectives.
        - Lander locations.
        - Rover movement capabilities (`can_traverse` combined with `visible`).
        - Rover equipment (`equipped_for_soil_analysis`, `equipped_for_rock_analysis`, `equipped_for_imaging`).
        - Store ownership (`store_of`).
        - Camera properties (`supports`, `calibration_target`, `on_board`).
        - Objective visibility (`visible_from`).
        - Waypoints visible from lander locations (`comm_wps`).
    - Precomputes shortest path travel costs between all pairs of waypoints for each rover using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Extract dynamic facts from the current state for quick lookup (rover locations, collected samples/images, sample locations, calibrated cameras, full stores).
    3. For each goal predicate in the task's goals:
        a. If the goal predicate is already true in the current state, continue to the next goal.
        b. If the goal is `(communicated_soil_data ?w)`:
            i. Find the target waypoint `w`.
            ii. Initialize minimum cost for this goal (`min_cost_g`) to infinity.
            iii. Iterate through each rover `r` equipped for soil analysis:
                - Get rover `r`'s current location (`r_loc`).
                - Initialize cost for this rover (`cost_r`) to 0.
                - Determine rover's location after potentially getting the sample (`r_loc_after_sample`). Initially, this is `r_loc`.
                - If rover `r` does not have `(have_soil_analysis ?r ?w)`:
                    - Check if `(at_soil_sample ?w)` is in the current state. If not, this rover cannot sample it now; skip this rover for this goal.
                    - Calculate travel cost from `r_loc` to `w`. If unreachable, skip rover.
                    - Add travel cost + 1 (for `sample_soil` action) to `cost_r`.
                    - Set `r_loc_after_sample` to `w`.
                    - (Ignored for simplicity: Add cost for `drop` if rover's store is full).
                - Calculate minimum travel cost from `r_loc_after_sample` to any communication waypoint (`comm_wp`). If unreachable, skip rover.
                - Add minimum communication travel cost + 1 (for `communicate_soil_data` action) to `cost_r`.
                - Update `min_cost_g = min(min_cost_g, cost_r)`.
            iv. If `min_cost_g` is still infinity, the goal is unreachable by any equipped rover from the current state; return a large value (e.g., 1000) immediately.
            v. Otherwise, add `min_cost_g` to the total heuristic cost.
        c. If the goal is `(communicated_rock_data ?w)`: Similar logic as soil data, using rock-specific predicates and equipped rovers.
        d. If the goal is `(communicated_image_data ?o ?m)`:
            i. Find the target objective `o` and mode `m`.
            ii. Initialize minimum cost for this goal (`min_cost_g`) to infinity.
            iii. Iterate through each rover `r` equipped for imaging:
                - Get rover `r`'s current location (`r_loc`).
                - Initialize cost for this rover (`cost_r`) to 0.
                - Determine rover's location after potentially taking the image (`r_loc_after_image`). Initially, this is `r_loc`.
                - If rover `r` does not have `(have_image ?r ?o ?m)`:
                    - Find suitable camera(s) on `r` supporting `m`. If none, skip rover.
                    - Find imaging waypoints for `o`. If none, skip rover.
                    - Determine if calibration is needed for any suitable camera. Find calibration target(s) and calibration waypoint(s). If calibration is needed but no calibration waypoint exists, skip rover.
                    - Calculate the minimum cost to get the image:
                        - If calibration is needed: min over cal_wp, img_wp: travel(r_loc, cal_wp) + 1 (calibrate) + travel(cal_wp, img_wp) + 1 (take_image). Track the `img_wp` that yields the minimum cost.
                        - If already calibrated: min over img_wp: travel(r_loc, img_wp) + 1 (take_image). Track the `img_wp` that yields the minimum cost.
                    - If minimum cost to get image is infinity, skip rover.
                    - Add minimum cost to get image to `cost_r`.
                    - Set `r_loc_after_image` to the tracked optimal `img_wp`.
                - Calculate minimum travel cost from `r_loc_after_image` to any communication waypoint (`comm_wp`). If unreachable, skip rover.
                - Add minimum communication travel cost + 1 (for `communicate_image_data` action) to `cost_r`.
                - Update `min_cost_g = min(min_cost_g, cost_r)`.
            iv. If `min_cost_g` is still infinity, the goal is unreachable; return 1000.
            v. Otherwise, add `min_cost_g` to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = set(task.static) # Use a set for fast lookup

        # Data structures to store parsed static information
        self.rovers = set()
        self.waypoints = set()
        self.landers = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()

        self.lander_wps = set()
        self.rover_movement_graphs = {} # {rover: {wp1: {wp2, ...}, ...}, ...}
        self.rovers_equipped_soil = set()
        self.rovers_equipped_rock = set()
        self.rovers_equipped_imaging = set()
        self.store_owner = {} # {store: rover}
        self.camera_supports_mode = {} # {camera: {mode, ...}}
        self.objective_visible_from = {} # {objective: {waypoint, ...}}
        self.camera_calibration_target = {} # {camera: objective}
        self.rover_cameras = {} # {rover: {camera, ...}}
        self.visible_wps_graph = {} # {wp1: {wp2, ...}, ...} # Needed for comm_wps

        # Parse static facts to populate data structures and identify objects
        all_facts_for_objects = set(task.static) | set(task.initial_state) # Include initial state to find all objects

        for fact in all_facts_for_objects:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts
            pred = parts[0]

            if pred == 'at_lander':
                lander, wp = parts[1], parts[2]
                self.landers.add(lander)
                self.waypoints.add(wp)
                self.lander_wps.add(wp)
            elif pred == 'at': # Rover location in initial state
                rover, wp = parts[1], parts[2]
                self.rovers.add(rover)
                self.waypoints.add(wp)
            elif pred == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                self.rovers.add(rover)
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
                # Movement graph edge exists if can_traverse AND visible
                # Check visible in static facts
                if f'(visible {wp1} {wp2})' in self.static_facts:
                     self.rover_movement_graphs.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)
            elif pred == 'visible':
                 wp1, wp2 = parts[1], parts[2]
                 self.waypoints.add(wp1)
                 self.waypoints.add(wp2)
                 self.visible_wps_graph.setdefault(wp1, set()).add(wp2)
            elif pred == 'equipped_for_soil_analysis': self.rovers.add(parts[1]); self.rovers_equipped_soil.add(parts[1])
            elif pred == 'equipped_for_rock_analysis': self.rovers.add(parts[1]); self.rovers_equipped_rock.add(parts[1])
            elif pred == 'equipped_for_imaging': self.rovers.add(parts[1]); self.rovers_equipped_imaging.add(parts[1])
            elif pred == 'store_of': store, rover = parts[1], parts[2]; self.stores.add(store); self.rovers.add(rover); self.store_owner[store] = rover
            elif pred == 'supports': camera, mode = parts[1], parts[2]; self.cameras.add(camera); self.modes.add(mode); self.camera_supports_mode.setdefault(camera, set()).add(mode)
            elif pred == 'visible_from': objective, wp = parts[1], parts[2]; self.objectives.add(objective); self.waypoints.add(wp); self.objective_visible_from.setdefault(objective, set()).add(wp)
            elif pred == 'calibration_target': camera, objective = parts[1], parts[2]; self.cameras.add(camera); self.objectives.add(objective); self.camera_calibration_target[camera] = objective
            elif pred == 'on_board': camera, rover = parts[1], parts[2]; self.cameras.add(camera); self.rovers.add(rover); self.rover_cameras.setdefault(rover, set()).add(camera)
            # Also add objects from dynamic predicates in initial state if not covered
            elif pred in ['at_soil_sample', 'at_rock_sample']: self.waypoints.add(parts[1])
            elif pred in ['empty', 'full']: self.stores.add(parts[1])
            elif pred in ['have_soil_analysis', 'have_rock_analysis']: self.rovers.add(parts[1]); self.waypoints.add(parts[2])
            elif pred == 'have_image': self.rovers.add(parts[1]); self.objectives.add(parts[2]); self.modes.add(parts[3])
            elif pred == 'calibrated': self.cameras.add(parts[1]); self.rovers.add(parts[2])


        # Precompute communication waypoints (visible from any lander waypoint)
        # Rover needs to be at x such that (visible x y) where y is lander_wp
        self.comm_wps = set()
        for lander_wp in self.lander_wps:
             for wp1, neighbors in self.visible_wps_graph.items():
                 if lander_wp in neighbors:
                     self.comm_wps.add(wp1)

        # Precompute travel costs for each rover
        self.rover_travel_costs = {}
        all_waypoints_list = list(self.waypoints) # Use a list for consistent indexing if needed, though dict is used here
        for rover in self.rovers:
            self.rover_travel_costs[rover] = self._bfs_all_pairs(rover, all_waypoints_list, self.rover_movement_graphs.get(rover, {}))

    def _bfs_all_pairs(self, rover, all_waypoints, movement_graph):
        """
        Computes shortest path costs from every reachable waypoint to every other
        reachable waypoint for a specific rover using BFS.
        """
        costs = {start_wp: {end_wp: math.inf for end_wp in all_waypoints} for start_wp in all_waypoints}

        for start_wp in all_waypoints:
            # Check if start_wp is part of the movement graph for this rover
            is_connected = start_wp in movement_graph or any(start_wp in neighbors for neighbors in movement_graph.values())
            if not is_connected:
                 continue # Rover cannot move from or to this waypoint

            costs[start_wp][start_wp] = 0
            queue = deque([start_wp])
            visited = {start_wp}

            while queue:
                curr_wp = queue.popleft()

                # Find neighbors: waypoints reachable from curr_wp by this rover
                neighbors = movement_graph.get(curr_wp, set())

                for next_wp in neighbors:
                    if next_wp not in visited:
                        visited.add(next_wp)
                        costs[start_wp][next_wp] = costs[start_wp][curr_wp] + 1
                        queue.append(next_wp)
        return costs

    def get_rover_location(self, rover, current_rover_locations):
        """Helper to get a rover's current waypoint from the state."""
        return current_rover_locations.get(rover)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Extract dynamic facts into sets/dicts for quick lookup
        current_rover_locations = {}
        current_have_soil = set() # {(r, w)}
        current_have_rock = set() # {(r, w)}
        current_have_image = set() # {(r, o, m)}
        current_at_soil_sample = set() # {w}
        current_at_rock_sample = set() # {w}
        current_calibrated = set() # {(c, r)}
        current_full_stores = set() # {s}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at': current_rover_locations[parts[1]] = parts[2]
            elif pred == 'have_soil_analysis': current_have_soil.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis': current_have_rock.add((parts[1], parts[2]))
            elif pred == 'have_image': current_have_image.add((parts[1], parts[2], parts[3]))
            elif pred == 'at_soil_sample': current_at_soil_sample.add(parts[1])
            elif pred == 'at_rock_sample': current_at_rock_sample.add(parts[1])
            elif pred == 'calibrated': current_calibrated.add((parts[1], parts[2]))
            elif pred == 'full': current_full_stores.add(parts[1])

        total_cost = 0
        unreachable_goal_cost = 1000 # Large value for unreachable goals

        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]

            min_cost_g = math.inf # Minimum cost for this specific goal predicate

            if pred == 'communicated_soil_data':
                w = parts[1]
                for r in self.rovers_equipped_soil:
                    r_loc = self.get_rover_location(r, current_rover_locations)
                    if r_loc is None: continue # Rover location unknown (shouldn't happen in valid states)

                    cost_r = 0
                    r_loc_after_sample = r_loc # Where rover is after sampling (if needed)

                    # Phase 1: Get sample if needed
                    if (r, w) not in current_have_soil:
                        # Need to sample. Check if sample is at the waypoint.
                        if w not in current_at_soil_sample:
                            # Sample not at waypoint w. This rover cannot sample it now.
                            # Check if *any* rover has the sample. If yes, this goal must be achieved by one of them.
                            # If no rover has it and it's not at W, the goal might be impossible via sampling.
                            # For this heuristic, if this rover doesn't have it and it's not at W, this rover cannot fulfill the 'get sample' part.
                            continue # Cannot get sample

                        # Sample is at waypoint w. Calculate travel cost.
                        travel_to_sample = self.rover_travel_costs[r].get(r_loc, {}).get(w, math.inf)
                        if travel_to_sample is math.inf: continue # Cannot reach sample point
                        cost_r += travel_to_sample + 1 # sample_soil action
                        r_loc_after_sample = w # Rover is now at waypoint w

                        # Add cost for drop if store is full? Ignore for simplicity.
                        # store = next((s for s, owner in self.store_owner.items() if owner == r), None)
                        # if store and store in current_full_stores: cost_r += 1 # drop action

                    # Phase 2: Communicate data
                    # Rover is at r_loc_after_sample. Need to reach a communication waypoint.
                    min_comm_travel = math.inf
                    for comm_wp in self.comm_wps:
                        travel = self.rover_travel_costs[r].get(r_loc_after_sample, {}).get(comm_wp, math.inf)
                        if travel is not math.inf:
                            min_comm_travel = min(min_comm_travel, travel)

                    if min_comm_travel is math.inf: continue # Cannot reach any communication point

                    cost_r += min_comm_travel + 1 # communicate_soil_data action

                    min_cost_g = min(min_cost_g, cost_r)

            elif pred == 'communicated_rock_data':
                w = parts[1]
                for r in self.rovers_equipped_rock:
                    r_loc = self.get_rover_location(r, current_rover_locations)
                    if r_loc is None: continue

                    cost_r = 0
                    r_loc_after_sample = r_loc

                    # Phase 1: Get sample if needed
                    if (r, w) not in current_have_rock:
                        if w not in current_at_rock_sample:
                             continue # Cannot get sample

                        travel_to_sample = self.rover_travel_costs[r].get(r_loc, {}).get(w, math.inf)
                        if travel_to_sample is math.inf: continue
                        cost_r += travel_to_sample + 1 # sample_rock action
                        r_loc_after_sample = w

                        # Add cost for drop if store is full? Ignore for simplicity.
                        # store = next((s for s, owner in self.store_owner.items() if owner == r), None)
                        # if store and store in current_full_stores: cost_r += 1 # drop

                    # Phase 2: Communicate data
                    min_comm_travel = math.inf
                    for comm_wp in self.comm_wps:
                        travel = self.rover_travel_costs[r].get(r_loc_after_sample, {}).get(comm_wp, math.inf)
                        if travel is not math.inf:
                            min_comm_travel = min(min_comm_travel, travel)

                    if min_comm_travel is math.inf: continue

                    cost_r += min_comm_travel + 1 # communicate_rock_data action

                    min_cost_g = min(min_cost_g, cost_r)

            elif pred == 'communicated_image_data':
                o, m = parts[1], parts[2]
                for r in self.rovers_equipped_imaging:
                    r_loc = self.get_rover_location(r, current_rover_locations)
                    if r_loc is None: continue

                    cost_r = 0
                    r_loc_after_image = r_loc # Where rover is after taking image (if needed)

                    # Phase 1: Get image if needed
                    if (r, o, m) not in current_have_image:
                        # Need to take image.
                        # Find suitable camera i on r supporting m.
                        suitable_cameras = [cam for cam in self.rover_cameras.get(r, []) if f'(supports {cam} {m})' in self.static_facts]
                        if not suitable_cameras: continue # No suitable camera on this rover

                        # Find imaging waypoints for o.
                        imaging_wps = self.objective_visible_from.get(o, [])
                        if not imaging_wps: continue # Cannot image this objective from anywhere

                        # Determine if calibration is needed for any suitable camera.
                        calibration_wps = set()
                        needs_calibration = False
                        for cam in suitable_cameras:
                            cal_target = self.camera_calibration_target.get(cam)
                            if cal_target:
                                cal_wps = self.objective_visible_from.get(cal_target, [])
                                calibration_wps.update(cal_wps)
                            if (cam, r) not in current_calibrated:
                                needs_calibration = True # At least one suitable camera needs calibration

                        if needs_calibration and not calibration_wps:
                             continue # Need calibration but no calibration waypoint found

                        # Calculate the minimum cost to get the image
                        min_cost_get_image = math.inf
                        best_img_wp = None # Track the imaging waypoint that gives min cost

                        if needs_calibration:
                            # Cost = travel(r_loc, cal_wp) + calibrate (1) + travel(cal_wp, img_wp) + take_image (1)
                            for cal_wp in calibration_wps:
                                travel1 = self.rover_travel_costs[r].get(r_loc, {}).get(cal_wp, math.inf)
                                if travel1 is math.inf: continue
                                for img_wp in imaging_wps:
                                    travel2 = self.rover_travel_costs[r].get(cal_wp, {}).get(img_wp, math.inf)
                                    if travel2 is math.inf: continue
                                    cost = travel1 + 1 + travel2 + 1
                                    if cost < min_cost_get_image:
                                        min_cost_get_image = cost
                                        best_img_wp = img_wp
                        else: # Already calibrated
                            # Cost = travel(r_loc, img_wp) + take_image (1)
                            for img_wp in imaging_wps:
                                travel = self.rover_travel_costs[r].get(r_loc, {}).get(img_wp, math.inf)
                                if travel is not math.inf:
                                    cost = travel + 1
                                    if cost < min_cost_get_image:
                                        min_cost_get_image = cost
                                        best_img_wp = img_wp

                        if min_cost_get_image is math.inf: continue # Cannot reach calibration or imaging point

                        cost_r += min_cost_get_image
                        r_loc_after_image = best_img_wp # Rover is now at the optimal imaging waypoint

                    # Phase 2: Communicate data
                    # Rover is at r_loc_after_image. Need to reach a communication waypoint.
                    min_comm_travel = math.inf
                    # If r_loc_after_image is None (e.g., rover already had image but its location wasn't found),
                    # use current location r_loc.
                    comm_start_loc = r_loc_after_image if r_loc_after_image is not None else r_loc

                    for comm_wp in self.comm_wps:
                        travel = self.rover_travel_costs[r].get(comm_start_loc, {}).get(comm_wp, math.inf)
                        if travel is not math.inf:
                            min_comm_travel = min(min_comm_travel, travel)

                    if min_comm_travel is math.inf: continue # Cannot reach any communication point

                    cost_r += min_comm_travel + 1 # communicate_image_data action

                    min_cost_g = min(min_cost_g, cost_r)

            # Add the minimum cost found for this goal predicate to the total cost
            if min_cost_g is not math.inf:
                total_cost += min_cost_g
            else:
                # If any goal is unreachable by any capable rover from the current state,
                # return a large value.
                return unreachable_goal_cost

        return total_cost
