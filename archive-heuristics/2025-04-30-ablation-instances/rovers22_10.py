# Required imports
from fnmatch import fnmatch
from collections import defaultdict
import sys # Used for float('inf')

# Assume Heuristic base class is available
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
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

# APSP implementation (Floyd-Warshall)
def floyd_warshall(waypoints, graph):
    """
    Computes all-pairs shortest paths using the Floyd-Warshall algorithm.

    Args:
        waypoints: A list or set of all waypoint names.
        graph: A dictionary representing the graph, where graph[u] is a set
               of waypoints directly reachable from u.

    Returns:
        A dictionary dist where dist[(u, v)] is the shortest path distance
        from waypoint u to waypoint v. Returns float('inf') if unreachable.
    """
    dist = {}
    for u in waypoints:
        for v in waypoints:
            if u == v:
                dist[(u, v)] = 0
            elif v in graph.get(u, set()):
                dist[(u, v)] = 1
            else:
                dist[(u, v)] = float('inf')

    for k in waypoints:
        for i in waypoints:
            for j in waypoints:
                dist[(i, j)] = min(dist[(i, j)], dist[(i, k) if (i, k) in dist else float('inf')] + dist[(k, j)] if (k, j) in dist else float('inf'))
                # Added checks for key existence in dist during min calculation for robustness

    return dist

# Helper to find min distance from a waypoint to any waypoint in a set
def min_dist_to_set(dist_matrix, start_wp, target_set):
    """
    Finds the minimum distance from start_wp to any waypoint in target_set.

    Args:
        dist_matrix: The result of floyd_warshall.
        start_wp: The starting waypoint.
        target_set: A set of target waypoints.

    Returns:
        The minimum distance, or float('inf') if no target is reachable or target_set is empty.
    """
    if not target_set or start_wp is None: # Cannot reach an empty set or from nowhere
        return float('inf')
    min_d = float('inf')
    for target_wp in target_set:
        min_d = min(min_d, dist_matrix.get((start_wp, target_wp), float('inf')))
    return min_d


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to satisfy
    all unsatisfied goal conditions related to communicating data (soil, rock, image).
    It sums the estimated cost for each individual unsatisfied communication goal.
    The cost for each goal is estimated based on whether the required data/image
    already exists or needs to be acquired first, plus the estimated navigation
    cost using precomputed shortest paths between waypoints.

    # Assumptions
    - All action costs are 1.
    - Navigation cost between waypoints is the shortest path length in the
      waypoint graph defined by the 'visible' predicate.
    - A single rover is responsible for acquiring and communicating a specific
      piece of data/image. The heuristic estimates the minimum cost over all
      suitable rovers.
    - If a soil/rock sample is no longer at a waypoint (`at_soil_sample`/`at_rock_sample` is false)
      but the goal requires communication from that waypoint, it is assumed
      that the data must already be held by a rover.
    - If a required object (like an equipped rover, camera, calibration target,
      or a waypoint visible from an objective/target) does not exist or is
      unreachable, the cost for that goal component is considered infinite.

    # Heuristic Initialization
    The heuristic precomputes static information from the task:
    - Lander locations.
    - Communication points (waypoints visible from any lander location).
    - Waypoint graph based on 'visible' predicates.
    - All-pairs shortest paths (APSP) between waypoints using Floyd-Warshall.
    - Which rovers are equipped for soil, rock, and imaging.
    - Which stores belong to which rovers.
    - Which cameras are on which rovers, which modes they support, and their calibration targets.
    - Which objectives are visible from which waypoints.
    - Which calibration targets are visible from which waypoints.
    - Precomputes minimum distances from any waypoint to sets of waypoints
      (communication points, waypoints visible from objectives/targets).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the sum of costs for each goal fact
    `(communicated_X_data ...)` that is not yet true in the state:

    1.  **Iterate through unsatisfied communication goals:** For each goal `g` in `self.goals`
        that is not present in the current `state`:
        a.  Add 1 to the total cost for the `communicate_X_data` action itself.
        b.  Determine the type of data (soil, rock, or image) and its parameters
            (waypoint `w` for soil/rock, objective `o` and mode `m` for image).

    2.  **Estimate cost for Soil/Rock Data (`communicated_soil_data ?w` or `communicated_rock_data ?w`):**
        a.  Check if `(have_soil_analysis ?r ?w)` (or `have_rock_analysis`) exists for any rover `r` in the current `state`.
        b.  If **data exists**:
            i.  Find all rovers `r` that have the data.
            ii. For each such rover, get its current location `curr`.
            iii. Calculate the navigation cost from `curr` to the nearest communication point (`min_dist_to_set(self.dist, curr, self.comm_points)`).
            iv. The minimum of these navigation costs over all rovers is added to the total cost.
        c.  If **data does NOT exist**:
            i.  Add 1 to the total cost for the `sample_soil` (or `sample_rock`) action.
            ii. Find all rovers equipped for the required analysis type.
            iii. For each equipped rover `r_equip`:
                - Get its current location `curr`.
                - Check if any of its stores are full. If yes, add 1 for the `drop` action.
                - Calculate the navigation cost from `curr` to the sample waypoint `w`, and then from `w` to the nearest communication point (`self.dist.get((curr, w), float('inf')) + self.min_dist_to_set(self.dist, w, self.comm_points)`).
                - The minimum of (drop_cost + navigation_cost) over all equipped rovers is added to the total cost.

    3.  **Estimate cost for Image Data (`communicated_image_data ?o ?m`):**
        a.  Check if `(have_image ?r ?o ?m)` exists for any rover `r` in the current `state`.
        b.  If **image exists**:
            i.  Find all rovers `r` that have the image.
            ii. For each such rover, get its current location `curr`.
            iii. Calculate the navigation cost from `curr` to the nearest communication point (`min_dist_to_set(self.dist, curr, self.comm_points)`).
            iv. The minimum of these navigation costs over all rovers is added to the total cost.
        c.  If **image does NOT exist**:
            i.  Add 1 to the total cost for the `take_image` action.
            ii. Find all equipped imaging rovers `r_equip` that have a camera `c` supporting mode `m`.
            iii. For each such (rover, camera) pair:
                - Get the rover's current location `curr`.
                - Check if the camera `c` is calibrated on rover `r_equip` in the current `state`.
                - If **camera is calibrated**:
                    - Find the minimum navigation cost from `curr` to a waypoint `p_view` visible from `o`, and then from that `p_view` to the nearest communication point (`min_dist_to_set(self.dist, curr, self.objective_visible_from.get(o, set())) + min_dist_to_set(self.dist, best_pview, self.comm_points)` where `best_pview` is the waypoint in `objective_visible_from[o]` closest to `curr`).
                - If **camera is NOT calibrated**:
                    - Add 1 for the `calibrate` action.
                    - Find the calibration target `t` for camera `c`.
                    - Find the minimum navigation cost from `curr` to a waypoint `p_cal` visible from `t`, then from that `p_cal` to a waypoint `p_view` visible from `o`, and finally from that `p_view` to the nearest communication point (`min_dist_to_set(self.dist, curr, self.caltarget_visible_from.get(t, set())) + min_dist_to_set(self.dist, best_pcal, self.objective_visible_from.get(o, set())) + min_dist_to_set(self.dist, best_pview, self.comm_points)` where `best_pcal` is closest to `curr` and `best_pview` is closest to `best_pcal`).
                - The minimum of these total costs (calibrate_cost + navigation_cost) over all candidate (rover, camera) pairs is added to the total cost.

    4.  **Return Total Cost:** The accumulated total cost is the heuristic value. If any required navigation or waypoint set is unreachable (cost is infinity), the total cost will be infinity, correctly indicating an unreachable goal component.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # --- Extract Static Information ---
        self.landers_at = {} # lander -> waypoint
        self.waypoint_graph = defaultdict(set) # waypoint -> set of visible waypoints
        self.waypoints = set()
        self.equipped_soil = set() # set of rovers
        self.equipped_rock = set() # set of rovers
        self.equipped_imaging = set() # set of rovers
        self.stores_of_rover = defaultdict(set) # rover -> set of stores
        self.camera_on_board = defaultdict(set) # rover -> set of cameras
        self.camera_supports = defaultdict(set) # camera -> set of modes
        self.camera_cal_target = {} # camera -> objective (calibration target)
        self.objective_visible_from = defaultdict(set) # objective -> set of waypoints
        self.caltarget_visible_from = defaultdict(set) # objective (cal target) -> set of waypoints

        # Collect all waypoints mentioned in static facts first
        # This is a heuristic way to find waypoints without parsing types
        potential_waypoints = set()
        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue
             # Waypoints appear in at_lander, visible, can_traverse, visible_from
             if len(parts) > 1:
                 for part in parts[1:]:
                     potential_waypoints.add(part)

        # Filter potential waypoints based on appearance in relevant predicates
        # This is still heuristic but slightly better than just checking prefix
        relevant_predicates_with_waypoints = ["at_lander", "visible", "can_traverse", "visible_from"]
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate in relevant_predicates_with_waypoints:
                 if len(parts) > 1:
                     for part in parts[1:]:
                         if part in potential_waypoints: # Check if it was seen as a potential waypoint object
                             self.waypoints.add(part)


        # Now process facts and build structures using identified waypoints
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at_lander":
                lander, wp = parts[1], parts[2]
                if wp in self.waypoints: self.landers_at[lander] = wp
            elif predicate == "visible":
                wp1, wp2 = parts[1], parts[2]
                if wp1 in self.waypoints and wp2 in self.waypoints:
                    self.waypoint_graph[wp1].add(wp2)
                    self.waypoint_graph[wp2].add(wp1) # Assuming visibility is symmetric for navigation
            elif predicate == "equipped_for_soil_analysis":
                self.equipped_soil.add(parts[1])
            elif predicate == "equipped_for_rock_analysis":
                self.equipped_rock.add(parts[1])
            elif predicate == "equipped_for_imaging":
                self.equipped_imaging.add(parts[1])
            elif predicate == "store_of":
                store, rover = parts[1], parts[2]
                self.stores_of_rover[rover].add(store)
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.camera_on_board[rover].add(camera)
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_supports[camera].add(mode)
            elif predicate == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.camera_cal_target[camera] = objective
            elif predicate == "visible_from":
                objective, wp = parts[1], parts[2]
                if wp in self.waypoints: self.objective_visible_from[objective].add(wp)

        # Build caltarget_visible_from based on camera_cal_target and objective_visible_from
        for cam, target_obj in self.camera_cal_target.items():
             if target_obj in self.objective_visible_from:
                 self.caltarget_visible_from[target_obj].update(self.objective_visible_from[target_obj])

        # Compute All-Pairs Shortest Paths (APSP)
        self.dist = floyd_warshall(list(self.waypoints), self.waypoint_graph)

        # Identify Communication Points
        self.comm_points = set()
        lander_locations = set(self.landers_at.values())
        for wp1 in self.waypoints:
            for wp2 in lander_locations:
                # Check if wp2 is visible from wp1
                if wp2 in self.waypoint_graph.get(wp1, set()):
                     self.comm_points.add(wp1)
                     break # Found a comm point for wp1, move to next wp1

        # Precompute min distances from any waypoint to relevant sets for efficiency
        self._min_dist_cache = {} # (start_wp, frozenset(target_set)) -> min_dist

        # Cache min dist to comm points
        comm_points_frozen = frozenset(self.comm_points)
        for wp in self.waypoints:
             self._min_dist_cache[(wp, comm_points_frozen)] = min_dist_to_set(self.dist, wp, self.comm_points)

        # Cache min dist to objective view points
        for obj, wps in self.objective_visible_from.items():
             wps_frozen = frozenset(wps)
             for wp in self.waypoints:
                 self._min_dist_cache[(wp, wps_frozen)] = min_dist_to_set(self.dist, wp, wps)

        # Cache min dist to cal target view points
        for target_obj, wps in self.caltarget_visible_from.items():
             wps_frozen = frozenset(wps)
             for wp in self.waypoints:
                 self._min_dist_cache[(wp, wps_frozen)] = min_dist_to_set(self.dist, wp, wps)


    def min_dist_to_set_cached(self, start_wp, target_set):
        """Helper to get cached min distance from start_wp to any waypoint in target_set."""
        if not target_set or start_wp is None:
             return float('inf')
        target_set_frozen = frozenset(target_set)
        # Use .get to return inf if the key is not in the cache (e.g., target_set was empty during init)
        return self._min_dist_cache.get((start_wp, target_set_frozen), float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        total_cost = 0  # Initialize action cost counter.

        # Get current rover locations for quick lookup
        rover_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                 rover, wp = parts[1], parts[2]
                 rover_locations[rover] = wp

        # Get full stores for quick lookup
        full_stores = {get_parts(fact)[1] for fact in state if match(fact, "full", "*")}

        # Get calibrated cameras for quick lookup
        calibrated_cameras = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "calibrated", "*", "*")}


        # Iterate through each goal fact
        for goal in self.goals:
            if goal in state:
                continue # Goal already satisfied

            parts = get_parts(goal)
            if not parts or parts[0] not in ["communicated_soil_data", "communicated_rock_data", "communicated_image_data"]:
                 # Ignore goals that are not communication goals or are malformed
                 continue

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                w = parts[1]
                goal_cost = 1 # Cost for communicate action

                # Check if data exists
                rovers_with_data = [r for r in rover_locations if match(f"(have_soil_analysis {r} {w})", state)]

                if rovers_with_data:
                    # Data exists, need to communicate
                    min_nav_cost = float('inf')
                    for r in rovers_with_data:
                        curr = rover_locations.get(r) # Use .get in case rover somehow isn't 'at' anywhere
                        nav_cost = self.min_dist_to_set_cached(curr, self.comm_points)
                        min_nav_cost = min(min_nav_cost, nav_cost)
                    goal_cost += min_nav_cost if min_nav_cost != float('inf') else 0
                else:
                    # Data doesn't exist, need to sample and communicate
                    goal_cost += 1 # Cost for sample action
                    min_acquisition_comm_cost = float('inf')

                    # Find equipped soil rovers that are currently at a known location
                    candidate_rovers = [r for r in self.equipped_soil if r in rover_locations]

                    for r_equip in candidate_rovers:
                        curr = rover_locations.get(r_equip)
                        if curr is None: continue # Skip if rover location is unknown

                        # Check if any store of this rover is full
                        store_full = any(s in full_stores for s in self.stores_of_rover.get(r_equip, []))
                        drop_cost = 1 if store_full else 0

                        # Nav cost: curr -> w -> comm_wp
                        dist_curr_to_w = self.dist.get((curr, w), float('inf'))
                        dist_w_to_comm = self.min_dist_to_set_cached(w, self.comm_points)

                        nav_cost = dist_curr_to_w + dist_w_to_comm
                        if nav_cost != float('inf'):
                             min_acquisition_comm_cost = min(min_acquisition_comm_cost, drop_cost + nav_cost)

                    goal_cost += min_acquisition_comm_cost if min_acquisition_comm_cost != float('inf') else 0

                total_cost += goal_cost

            elif predicate == "communicated_rock_data":
                w = parts[1]
                goal_cost = 1 # Cost for communicate action

                # Check if data exists
                rovers_with_data = [r for r in rover_locations if match(f"(have_rock_analysis {r} {w})", state)]

                if rovers_with_data:
                    # Data exists, need to communicate
                    min_nav_cost = float('inf')
                    for r in rovers_with_data:
                        curr = rover_locations.get(r)
                        nav_cost = self.min_dist_to_set_cached(curr, self.comm_points)
                        min_nav_cost = min(min_nav_cost, nav_cost)
                    goal_cost += min_nav_cost if min_nav_cost != float('inf') else 0
                else:
                    # Data doesn't exist, need to sample and communicate
                    goal_cost += 1 # Cost for sample action
                    min_acquisition_comm_cost = float('inf')

                    # Find equipped rock rovers that are currently at a known location
                    candidate_rovers = [r for r in self.equipped_rock if r in rover_locations]

                    for r_equip in candidate_rovers:
                        curr = rover_locations.get(r_equip)
                        if curr is None: continue

                        # Check if any store of this rover is full
                        store_full = any(s in full_stores for s in self.stores_of_rover.get(r_equip, []))
                        drop_cost = 1 if store_full else 0

                        # Nav cost: curr -> w -> comm_wp
                        dist_curr_to_w = self.dist.get((curr, w), float('inf'))
                        dist_w_to_comm = self.min_dist_to_set_cached(w, self.comm_points)

                        nav_cost = dist_curr_to_w + dist_w_to_comm
                        if nav_cost != float('inf'):
                             min_acquisition_comm_cost = min(min_acquisition_comm_cost, drop_cost + nav_cost)

                    goal_cost += min_acquisition_comm_cost if min_acquisition_comm_cost != float('inf') else 0

                total_cost += goal_cost

            elif predicate == "communicated_image_data":
                o, m = parts[1], parts[2]
                goal_cost = 1 # Cost for communicate action

                # Check if image exists
                rovers_with_image = [r for r in rover_locations if match(f"(have_image {r} {o} {m})", state)]

                if rovers_with_image:
                    # Image exists, need to communicate
                    min_nav_cost = float('inf')
                    for r in rovers_with_image:
                        curr = rover_locations.get(r)
                        nav_cost = self.min_dist_to_set_cached(curr, self.comm_points)
                        min_nav_cost = min(min_nav_cost, nav_cost)
                    goal_cost += min_nav_cost if min_nav_cost != float('inf') else 0
                else:
                    # Image doesn't exist, need to take image and communicate
                    goal_cost += 1 # Cost for take_image action
                    min_acquisition_comm_cost = float('inf')

                    # Find equipped imaging rovers with suitable cameras that are currently at a known location
                    candidate_rovers_cameras = []
                    for r_equip in self.equipped_imaging:
                        if r_equip in rover_locations:
                            for cam in self.camera_on_board.get(r_equip, []):
                                if m in self.camera_supports.get(cam, []):
                                    candidate_rovers_cameras.append((r_equip, cam))

                    for r_equip, cam in candidate_rovers_cameras:
                        curr = rover_locations.get(r_equip)
                        if curr is None: continue

                        calibrated = (cam, r_equip) in calibrated_cameras

                        if calibrated:
                            # Need to visit p_view, then comm_wp
                            obj_view_points = self.objective_visible_from.get(o, set())
                            if not obj_view_points: continue # Cannot view objective

                            # Find the best p_view: closest to curr
                            min_dist_curr_to_pview = float('inf')
                            best_pview = None
                            for p_view in obj_view_points:
                                d = self.dist.get((curr, p_view), float('inf'))
                                if d < min_dist_curr_to_pview:
                                    min_dist_curr_to_pview = d
                                    best_pview = p_view

                            if best_pview is not None and min_dist_curr_to_pview != float('inf'):
                                dist_pview_to_comm = self.min_dist_to_set_cached(best_pview, self.comm_points)
                                nav_cost = min_dist_curr_to_pview + dist_pview_to_comm
                                if nav_cost != float('inf'):
                                    min_acquisition_comm_cost = min(min_acquisition_comm_cost, nav_cost)

                        else:
                            # Need to calibrate, then visit p_view, then comm_wp
                            cost_calibrate = 1 # Cost for calibrate action
                            cal_target = self.camera_cal_target.get(cam)
                            if cal_target is None: continue # Cannot calibrate this camera

                            cal_view_points = self.caltarget_visible_from.get(cal_target, set())
                            if not cal_view_points: continue # Cannot view calibration target

                            obj_view_points = self.objective_visible_from.get(o, set())
                            if not obj_view_points: continue # Cannot view objective

                            # Find the best p_cal: closest to curr
                            min_dist_curr_to_pcal = float('inf')
                            best_pcal = None
                            for p_cal in cal_view_points:
                                 d = self.dist.get((curr, p_cal), float('inf'))
                                 if d < min_dist_curr_to_pcal:
                                     min_dist_curr_to_pcal = d
                                     best_pcal = p_cal

                            if best_pcal is not None and min_dist_curr_to_pcal != float('inf'):
                                # Find the best p_view: closest to best_pcal
                                min_dist_pcal_to_pview = float('inf')
                                best_pview = None
                                for p_view in obj_view_points:
                                    d = self.dist.get((best_pcal, p_view), float('inf'))
                                    if d < min_dist_pcal_to_pview:
                                        min_dist_pcal_to_pview = d
                                        best_pview = p_view

                                if best_pview is not None and min_dist_pcal_to_pview != float('inf'):
                                    dist_pview_to_comm = self.min_dist_to_set_cached(best_pview, self.comm_points)
                                    nav_cost = min_dist_curr_to_pcal + min_dist_pcal_to_pview + dist_pview_to_comm
                                    if nav_cost != float('inf'):
                                        min_acquisition_comm_cost = min(min_acquisition_comm_cost, cost_calibrate + nav_cost)

                    goal_cost += min_acquisition_comm_cost if min_acquisition_comm_cost != float('inf') else 0

                total_cost += goal_cost

        return total_cost
