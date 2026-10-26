from collections import deque
import math
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at rover1 waypoint1)" -> ["at", "rover1", "waypoint1"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts must match the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all goal
    conditions. It decomposes the problem into achieving the necessary
    intermediate states (collecting samples, taking images) and then
    communicating the data. It sums the estimated minimum costs for each
    item that needs to be collected/taken, plus the cost for communication.
    Communication cost includes one navigation trip to a communication point
    (if any communication is needed) plus one action per item to be communicated.
    Navigation costs are estimated using precomputed shortest paths on the
    waypoint graph.

    # Assumptions
    - All goal facts are of the form (communicated_soil_data ?w),
      (communicated_rock_data ?w), or (communicated_image_data ?o ?m).
    - The waypoint graph defined by (visible ?w1 ?w2) is symmetric and
      traversable by all rovers if (can_traverse ?r ?w1 ?w2) exists for any r.
      (Simplified: using 'visible' facts to build a single symmetric graph).
    - Samples (soil/rock) at waypoints, once sampled, are removed from the ground
      and cannot be resampled. If a sample was originally present but is no longer
      on the ground and no rover has the data, the goal is unreachable.
    - Dropping a sample frees the store but the rover retains the 'have_soil_analysis'
      or 'have_rock_analysis' fact.
    - Calibration is consumed by the take_image action.
    - Any rover can communicate any data it possesses from a communication point.

    # Heuristic Initialization
    - Extracts goal facts.
    - Extracts static facts: lander location, rover capabilities, camera info
      (on-board, supports, calibration target), objective viewpoints, store ownership.
    - Extracts initial state facts to identify initial sample locations.
    - Builds the waypoint graph based on 'visible' facts (assuming symmetric).
    - Computes all-pairs shortest paths between waypoints using BFS.
    - Identifies communication waypoints (visible *to* lander location).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal facts that are not satisfied in the current state.
    2. Identify which soil samples, rock samples, and images are *already* collected/taken by any rover in the current state.
    3. Initialize heuristic value `h = 0`.
    4. Identify items that need collection/taking:
       - For each `(communicated_soil_data w)` goal not in state: If `w` is not collected, add `w` to a list of needed soil collections.
       - For each `(communicated_rock_data w)` goal not in state: If `w` is not collected, add `w` to a list of needed rock collections.
       - For each `(communicated_image_data o m)` goal not in state: If `(o, m)` is not taken, add `(o, m)` to a list of needed image takings.
    5. For each item in the needed collection/taking lists:
       - Estimate the minimum cost for *any* suitable rover to perform the collection/taking action (including navigation, potential drop, potential calibration). Add this minimum cost to `h`. If impossible, return infinity.
    6. Identify items that need communication: These are all items corresponding to the unsatisfied goal facts.
    7. Add the number of distinct items needing communication to `h` (this counts the `communicate_X_data` actions).
    8. If there are items needing communication, estimate the minimum navigation cost for *any* rover to reach *any* communication waypoint from its current location. Add this minimum navigation cost to `h`. If no rover can reach a comm point, return infinity.
    9. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static/initial facts."""
        self.goals = set(task.goals)
        self.initial_facts = set(task.init)
        static_facts = set(task.static)

        # --- Extract Static Information ---
        self.lander_location = None
        self.rover_capabilities = {} # rover -> set of capabilities ('soil', 'rock', 'imaging')
        self.rover_stores = {}       # rover -> store
        self.rover_cameras = {}      # rover -> set of cameras
        self.camera_modes = {}       # camera -> set of modes
        self.camera_calibration_target = {} # camera -> objective
        self.objective_viewpoints = {} # objective -> set of waypoints
        self.waypoint_graph = {}     # waypoint -> set of connected waypoints (based on visible)
        self.initial_soil_samples = set() # waypoint
        self.initial_rock_samples = set() # waypoint
        self.all_waypoints = set()

        for fact in self.initial_facts:
             if match(fact, "at_lander", "*", "*"):
                 self.lander_location = get_parts(fact)[2]
             if match(fact, "at_soil_sample", "*"):
                 self.initial_soil_samples.add(get_parts(fact)[1])
             if match(fact, "at_rock_sample", "*"):
                 self.initial_rock_samples.add(get_parts(fact)[1])
             if match(fact, "at", "*", "*"): # Collect all initial waypoints
                 self.all_waypoints.add(get_parts(fact)[2])

        for fact in static_facts:
            parts = get_parts(fact)
            if match(fact, "equipped_for_soil_analysis", "*"):
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('soil')
            elif match(fact, "equipped_for_rock_analysis", "*"):
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('rock')
            elif match(fact, "equipped_for_imaging", "*"):
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('imaging')
            elif match(fact, "store_of", "*", "*"):
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
            elif match(fact, "on_board", "*", "*"):
                camera, rover = parts[1], parts[2]
                self.rover_cameras.setdefault(rover, set()).add(camera)
            elif match(fact, "supports", "*", "*"):
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = parts[1], parts[2]
                self.camera_calibration_target[camera] = objective
            elif match(fact, "visible_from", "*", "*"):
                objective, waypoint = parts[1], parts[2]
                self.objective_viewpoints.setdefault(objective, set()).add(waypoint)
                self.all_waypoints.add(waypoint) # Collect all waypoints mentioned
            elif match(fact, "visible", "*", "*"):
                 w1, w2 = parts[1], parts[2]
                 self.waypoint_graph.setdefault(w1, set()).add(w2)
                 self.waypoint_graph.setdefault(w2, set()).add(w1) # Assume symmetric
                 self.all_waypoints.add(w1)
                 self.all_waypoints.add(w2)
            # Assuming can_traverse facts are consistent with visible for graph

        # Ensure all waypoints mentioned in init/static are in the graph keys
        for w in self.all_waypoints:
             self.waypoint_graph.setdefault(w, set())

        # --- Precompute Shortest Paths (BFS from each waypoint) ---
        self.dist_matrix = {} # (w1, w2) -> distance
        for start_node in self.waypoint_graph:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.dist_matrix[(start_node, start_node)] = 0

            while q:
                current_node, dist = q.popleft()

                for neighbor in self.waypoint_graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.dist_matrix[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

        # --- Identify Communication Waypoints ---
        self.comm_waypoints = set()
        if self.lander_location:
            # Comm waypoints are those ?x such that (visible ?x lander_location).
            for fact in static_facts:
                 if match(fact, "visible", "*", self.lander_location):
                      self.comm_waypoints.add(get_parts(fact)[1])


    def get_shortest_dist(self, w1, w2):
        """Get shortest distance between two waypoints."""
        return self.dist_matrix.get((w1, w2), math.inf)

    def get_nearest_comm_waypoint_dist(self, w):
        """Get shortest distance from waypoint w to any communication waypoint."""
        min_dist = math.inf
        if not self.comm_waypoints:
             return math.inf # No communication points exist

        for comm_w in self.comm_waypoints:
            min_dist = min(min_dist, self.get_shortest_dist(w, comm_w))
        return min_dist

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = set(node.state) # Convert frozenset to set for easier lookup

        # --- Parse Current State ---
        rover_locations = {}
        rover_soil_samples = {} # rover -> set of waypoints
        rover_rock_samples = {} # rover -> set of waypoints
        rover_images = {}       # rover -> set of (objective, mode)
        calibrated_cameras = set() # (camera, rover)
        store_status = {}       # store -> 'empty' or 'full'
        current_at_soil_sample = set() # waypoint
        current_at_rock_sample = set() # waypoint

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                rover, waypoint = parts[1], parts[2]
                rover_locations[rover] = waypoint
            elif match(fact, "have_soil_analysis", "*", "*"):
                rover, waypoint = parts[1], parts[2]
                rover_soil_samples.setdefault(rover, set()).add(waypoint)
            elif match(fact, "have_rock_analysis", "*", "*"):
                rover, waypoint = parts[1], parts[2]
                rover_rock_samples.setdefault(rover, set()).add(waypoint)
            elif match(fact, "have_image", "*", "*", "*"):
                rover, objective, mode = parts[1], parts[2], parts[3]
                rover_images.setdefault(rover, set()).add((objective, mode))
            elif match(fact, "calibrated", "*", "*"):
                camera, rover = parts[1], parts[2]
                calibrated_cameras.add((camera, rover))
            elif match(fact, "empty", "*"):
                store = parts[1]
                store_status[store] = 'empty'
            elif match(fact, "full", "*"):
                store = parts[1]
                store_status[store] = 'full'
            elif match(fact, "at_soil_sample", "*"):
                 current_at_soil_sample.add(parts[1])
            elif match(fact, "at_rock_sample", "*"):
                 current_at_rock_sample.add(parts[1])

        # Ensure all rovers are in rover_locations, default to None or a dummy if not found (shouldn't happen in valid states)
        for rover in self.rover_capabilities:
             if rover not in rover_locations:
                  # This state is likely unreachable or invalid, return inf
                  return math.inf


        # --- Compute Heuristic ---
        h = 0
        unsatisfied_goals = self.goals - state

        # Sets to track items already collected/taken across all rovers
        collected_soil_waypoints = {w for samples in rover_soil_samples.values() for w in samples}
        collected_rock_waypoints = {w for samples in rover_rock_samples.values() for w in samples}
        taken_images = {img for images in rover_images.values() for img in images}

        items_needing_communication = set()

        for goal in unsatisfied_goals:
            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                w = parts[1]
                items_needing_communication.add(('soil', w))

                # Cost to collect (if needed)
                if w not in collected_soil_waypoints:
                    # Check if sample is available on the ground initially
                    if w not in self.initial_soil_samples:
                         return math.inf # Sample never existed

                    # Check if sample is still on the ground
                    if w not in current_at_soil_sample:
                         # Sample was here initially, but is gone from ground and not collected by any rover.
                         # This implies an issue or unsolvability.
                         return math.inf

                    min_collect_cost = math.inf
                    for rover, capabilities in self.rover_capabilities.items():
                        if 'soil' in capabilities:
                            loc_r = rover_locations[rover]
                            store = self.rover_stores.get(rover)
                            store_full = (store_status.get(store) == 'full') if store else False # Assume not full if no store defined
                            dist_to_sample = self.get_shortest_dist(loc_r, w)
                            if dist_to_sample != math.inf:
                                cost = dist_to_sample + (1 if store_full) + 1 # navigate + (drop) + sample
                                min_collect_cost = min(min_collect_cost, cost)
                    if min_collect_cost == math.inf: return math.inf # Cannot collect
                    h += min_collect_cost


            elif predicate == 'communicated_rock_data':
                w = parts[1]
                items_needing_communication.add(('rock', w))

                # Cost to collect (if needed)
                if w not in collected_rock_waypoints:
                    # Check if sample is available on the ground initially
                    if w not in self.initial_rock_samples:
                         return math.inf # Sample never existed

                    # Check if sample is still on the ground
                    if w not in current_at_rock_sample:
                         return math.inf # Sample gone from ground and not collected

                    min_collect_cost = math.inf
                    for rover, capabilities in self.rover_capabilities.items():
                        if 'rock' in capabilities:
                            loc_r = rover_locations[rover]
                            store = self.rover_stores.get(rover)
                            store_full = (store_status.get(store) == 'full') if store else False
                            dist_to_sample = self.get_shortest_dist(loc_r, w)
                            if dist_to_sample != math.inf:
                                cost = dist_to_sample + (1 if store_full) + 1 # navigate + (drop) + sample
                                min_collect_cost = min(min_collect_cost, cost)
                    if min_collect_cost == math.inf: return math.inf # Cannot collect
                    h += min_collect_cost


            elif predicate == 'communicated_image_data':
                o, m = parts[1], parts[2]
                image_key = (o, m)
                items_needing_communication.add(('image', o, m))

                # Cost to take image (if needed)
                if image_key not in taken_images:
                    min_take_cost = math.inf
                    # Find suitable rover, camera, viewpoint
                    for rover, capabilities in self.rover_capabilities.items():
                        if 'imaging' in capabilities and rover in rover_locations:
                            loc_r = rover_locations[rover]
                            for camera in self.rover_cameras.get(rover, set()):
                                if m in self.camera_modes.get(camera, set()):
                                    for p in self.objective_viewpoints.get(o, set()):
                                        is_calibrated = (camera, rover) in calibrated_cameras
                                        cost_to_viewpoint = self.get_shortest_dist(loc_r, p)

                                        if cost_to_viewpoint != math.inf:
                                            if is_calibrated:
                                                # Already calibrated, just need to get to viewpoint and take image
                                                cost = cost_to_viewpoint + 1 # navigate + take_image
                                                min_take_cost = min(min_take_cost, cost)
                                            else:
                                                # Needs calibration first
                                                cal_target = self.camera_calibration_target.get(camera)
                                                if cal_target:
                                                    cal_waypoints = self.objective_viewpoints.get(cal_target, set())
                                                    min_cal_sequence_cost = math.inf
                                                    for cal_w in cal_waypoints:
                                                        cost_to_cal = self.get_shortest_dist(loc_r, cal_w)
                                                        cost_cal_to_viewpoint = self.get_shortest_dist(cal_w, p)
                                                        if cost_to_cal != math.inf and cost_cal_to_viewpoint != math.inf:
                                                            # navigate to cal_w + calibrate + navigate to p + take_image
                                                            cal_sequence_cost = cost_to_cal + 1 + cost_cal_to_viewpoint + 1
                                                            min_cal_sequence_cost = min(min_cal_sequence_cost, cal_sequence_cost)
                                                    if min_cal_sequence_cost != math.inf:
                                                         min_take_cost = min(min_take_cost, min_cal_sequence_cost)

                    if min_take_cost == math.inf: return math.inf # Cannot take image
                    h += min_take_cost

        # Cost to communicate (if needed)
        if items_needing_communication:
             # Add cost for the communication actions themselves
             h += len(items_needing_communication)

             # Add cost for navigation to a communication point (once per planning horizon)
             min_rover_dist_to_comm = math.inf
             for rover, loc_r in rover_locations.items():
                  dist_to_comm = self.get_nearest_comm_waypoint_dist(loc_r)
                  if dist_to_comm != math.inf:
                       min_rover_dist_to_comm = min(min_rover_dist_to_comm, dist_to_comm)

             if min_rover_dist_to_comm == math.inf: return math.inf # Cannot reach comm point
             h += min_rover_dist_to_comm # navigate to comm point

        return h
