from fnmatch import fnmatch
import collections
import math

# Assume Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if it's not provided in the execution environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or invalid format gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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


def bfs(graph, start_node, nodes):
    """Computes shortest distances from start_node to all other nodes using BFS."""
    distances = {node: float('inf') for node in nodes}
    if start_node not in nodes:
         return distances # Start node not in the known graph nodes

    distances[start_node] = 0
    queue = collections.deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor in nodes and neighbor not in visited: # Ensure neighbor is a known node
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


def compute_all_pairs_shortest_paths(graph, nodes):
    """Computes shortest distances between all pairs of nodes."""
    all_distances = {}
    for start_node in nodes:
        all_distances[start_node] = bfs(graph, start_node, nodes)
    return all_distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing up the estimated costs
    for each unachieved goal fact. The cost for each goal fact is estimated
    independently, considering the necessary steps (sampling/imaging,
    calibration, navigation, communication) and finding the minimum cost
    path and required actions for the most capable/best-positioned rover.
    Navigation costs are based on precomputed shortest paths.
    """

    UNREACHABLE_COST = 1000000  # Large value indicating an unreachable goal component

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts
        to precompute navigation distances and static relationships.
        """
        self.goals = task.goals

        # --- Precompute static information ---
        self.lander_location = None
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.store_to_rover = {}  # store -> rover
        self.rover_to_store = {}  # rover -> store
        self.waypoint_graph = {}  # waypoint -> set of connected waypoints
        self.all_waypoints = set()
        can_traverse_facts = set()
        self.calib_targets = {}  # camera -> objective (calibration target)
        self.camera_on_board = {}  # camera -> rover
        self.camera_supports = collections.defaultdict(set)  # camera -> set of modes
        self.objective_image_wps = collections.defaultdict(set)  # objective -> set of waypoints (for imaging)
        self.objective_calib_wps = collections.defaultdict(set)  # camera -> set of waypoints (for calibration)

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue

            if match(fact, "at_lander", "*", "*"):
                self.lander_location = parts[2]
                self.all_waypoints.add(self.lander_location)
            elif match(fact, "equipped_for_soil_analysis", "*"):
                self.equipped_soil.add(parts[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.equipped_rock.add(parts[1])
            elif match(fact, "equipped_for_imaging", "*"):
                self.equipped_imaging.add(parts[1])
            elif match(fact, "store_of", "*", "*"):
                store, rover = parts[1], parts[2]
                self.store_to_rover[store] = rover
                self.rover_to_store[rover] = store
            elif match(fact, "can_traverse", "*", "*", "*"):
                # Assuming can_traverse is symmetric and same for all rovers for navigation graph
                can_traverse_facts.add(fact)
                r, w1, w2 = parts[1], parts[2], parts[3]
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = parts[1], parts[2]
                self.calib_targets[camera] = objective
            elif match(fact, "on_board", "*", "*"):
                camera, rover = parts[1], parts[2]
                self.camera_on_board[camera] = rover
            elif match(fact, "supports", "*", "*"):
                camera, mode = parts[1], parts[2]
                self.camera_supports[camera].add(mode)
            elif match(fact, "visible_from", "*", "*"):
                objective, waypoint = parts[1], parts[2]
                self.all_waypoints.add(waypoint)
                # Check if this objective is a calibration target for any camera
                calibrating_cameras = [cam for cam, obj in self.calib_targets.items() if obj == objective]
                if calibrating_cameras:
                     for cam in calibrating_cameras:
                         self.objective_calib_wps[cam].add(waypoint)
                else:
                     # This is an image waypoint for the objective
                     self.objective_image_wps[objective].add(waypoint)
            elif match(fact, "visible", "*", "*"):
                 # Add waypoints from visible facts to ensure they are included
                 w1, w2 = parts[1], parts[2]
                 self.all_waypoints.add(w1)
                 self.all_waypoints.add(w2)


        # Build navigation graph from collected can_traverse facts
        # Assuming symmetric and same for all rovers
        for fact in can_traverse_facts:
            r, w1, w2 = get_parts(fact)[1:]
            self.waypoint_graph.setdefault(w1, set()).add(w2)
            self.waypoint_graph.setdefault(w2, set()).add(w1) # Assume symmetric

        # Compute all-pairs shortest paths
        self.waypoint_distances = compute_all_pairs_shortest_paths(self.waypoint_graph, list(self.all_waypoints))

        # Identify communication waypoints (visible from lander location)
        self.comm_wps = set()
        if self.lander_location:
            for fact in task.static:
                if match(fact, "visible", "*", self.lander_location):
                    self.comm_wps.add(get_parts(fact)[1])
                elif match(fact, "visible", self.lander_location, "*"):
                     # Assuming visible is symmetric, but check both directions
                     self.comm_wps.add(get_parts(fact)[2])

        # Parse initial state for initial samples (they are removed when sampled)
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        for fact in task.initial_state:
            if match(fact, "at_soil_sample", "*"):
                self.initial_soil_samples.add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                self.initial_rock_samples.add(get_parts(fact)[1])

    def _get_dist(self, w1, w2):
        """Get shortest distance between two waypoints."""
        if w1 not in self.waypoint_distances or w2 not in self.waypoint_distances[w1]:
            # This case indicates an issue if w1/w2 are in all_waypoints but not in distances map
            # Or if w1/w2 are not in all_waypoints at all (shouldn't happen if parsing is correct)
            return float('inf')
        return self.waypoint_distances[w1][w2]

    def _min_dist_to_comm_wp(self, start_wp):
        """Get minimum distance from start_wp to any communication waypoint."""
        if not self.comm_wps:
            return float('inf') # No communication points available
        min_dist = float('inf')
        for comm_wp in self.comm_wps:
            min_dist = min(min_dist, self._get_dist(start_wp, comm_wp))
        return min_dist

    def _cost_to_communicate_soil(self, w, state, rover_locs, store_status, have_soil):
        """Estimate cost for a single communicated_soil_data goal."""
        # Check if any rover already has the sample
        rovers_with_sample = {r for r, samples in have_soil.items() if w in samples}

        if rovers_with_sample:
            # Sample is held, just need to communicate
            min_comm_cost = float('inf')
            for r_has in rovers_with_sample:
                r_loc = rover_locs.get(r_has)
                if r_loc is None: continue # Rover location unknown
                comm_move_cost = self._min_dist_to_comm_wp(r_loc)
                if comm_move_cost != float('inf'):
                    min_comm_cost = min(min_comm_cost, comm_move_cost + 1) # +1 for communicate action
            return min_comm_cost if min_comm_cost != float('inf') else self.UNREACHABLE_COST
        else:
            # Sample is not held, need to sample it first
            # Check if the sample exists at the waypoint initially
            if w not in self.initial_soil_samples:
                # Sample never existed at this waypoint initially, impossible via sampling
                return self.UNREACHABLE_COST

            # Find capable rovers
            capable_rovers = self.equipped_soil
            if not capable_rovers:
                 return self.UNREACHABLE_COST # No rover can sample soil

            min_total_cost = float('inf')
            for r_eq in capable_rovers:
                r_loc = rover_locs.get(r_eq)
                if r_loc is None: continue # Rover location unknown

                # Cost to get the sample
                sample_move_cost = self._get_dist(r_loc, w)
                if sample_move_cost == float('inf'): continue # Cannot reach sample waypoint

                rover_store = self.rover_to_store.get(r_eq)
                drop_cost = 0
                if rover_store and store_status.get(rover_store) == 'full':
                     drop_cost = 1 # Need to drop before sampling

                sample_action_cost = 1

                # Cost to communicate after sampling (rover is at waypoint w)
                comm_move_cost = self._min_dist_to_comm_wp(w)
                if comm_move_cost == float('inf'): continue # Cannot reach communication waypoint from w

                total_cost = sample_move_cost + drop_cost + sample_action_cost + comm_move_cost + 1 # +1 communicate action
                min_total_cost = min(min_total_cost, total_cost)

            return min_total_cost if min_total_cost != float('inf') else self.UNREACHABLE_COST

    def _cost_to_communicate_rock(self, w, state, rover_locs, store_status, have_rock):
        """Estimate cost for a single communicated_rock_data goal."""
        # Check if any rover already has the sample
        rovers_with_sample = {r for r, samples in have_rock.items() if w in samples}

        if rovers_with_sample:
            # Sample is held, just need to communicate
            min_comm_cost = float('inf')
            for r_has in rovers_with_sample:
                r_loc = rover_locs.get(r_has)
                if r_loc is None: continue # Rover location unknown
                comm_move_cost = self._min_dist_to_comm_wp(r_loc)
                if comm_move_cost != float('inf'):
                    min_comm_cost = min(min_comm_cost, comm_move_cost + 1) # +1 for communicate action
            return min_comm_cost if min_comm_cost != float('inf') else self.UNREACHABLE_COST
        else:
            # Sample is not held, need to sample it first
            # Check if the sample exists at the waypoint initially
            if w not in self.initial_rock_samples:
                # Sample never existed at this waypoint initially, impossible via sampling
                return self.UNREACHABLE_COST

            # Find capable rovers
            capable_rovers = self.equipped_rock
            if not capable_rovers:
                 return self.UNREACHABLE_COST # No rover can sample rock

            min_total_cost = float('inf')
            for r_eq in capable_rovers:
                r_loc = rover_locs.get(r_eq)
                if r_loc is None: continue # Rover location unknown

                # Cost to get the sample
                sample_move_cost = self._get_dist(r_loc, w)
                if sample_move_cost == float('inf'): continue # Cannot reach sample waypoint

                rover_store = self.rover_to_store.get(r_eq)
                drop_cost = 0
                if rover_store and store_status.get(rover_store) == 'full':
                     drop_cost = 1 # Need to drop before sampling

                sample_action_cost = 1

                # Cost to communicate after sampling (rover is at waypoint w)
                comm_move_cost = self._min_dist_to_comm_wp(w)
                if comm_move_cost == float('inf'): continue # Cannot reach communication waypoint from w

                total_cost = sample_move_cost + drop_cost + sample_action_cost + comm_move_cost + 1 # +1 communicate action
                min_total_cost = min(min_total_cost, total_cost)

            return min_total_cost if min_total_cost != float('inf') else self.UNREACHABLE_COST


    def _cost_to_communicate_image(self, o, m, state, rover_locs, calibrated_cams, have_image):
        """Estimate cost for a single communicated_image_data goal."""
        # Check if any rover already has the image
        rovers_with_image = {r for r, obj, mode in have_image if obj == o and mode == m}

        if rovers_with_image:
            # Image is held, just need to communicate
            min_comm_cost = float('inf')
            for r_has in rovers_with_image:
                r_loc = rover_locs.get(r_has)
                if r_loc is None: continue # Rover location unknown
                comm_move_cost = self._min_dist_to_comm_wp(r_loc)
                if comm_move_cost != float('inf'):
                    min_comm_cost = min(min_comm_cost, comm_move_cost + 1) # +1 for communicate action
            return min_comm_cost if min_comm_cost != float('inf') else self.UNREACHABLE_COST
        else:
            # Image is not held, need to take it first
            # Find capable (rover, camera) pairs
            capable_ri_pairs = set()
            for camera, rover in self.camera_on_board.items():
                if rover in self.equipped_imaging and m in self.camera_supports.get(camera, set()):
                    capable_ri_pairs.add((rover, camera))

            if not capable_ri_pairs:
                return self.UNREACHABLE_COST # No rover/camera can take this image

            # Find image waypoints for this objective
            image_wps_o = self.objective_image_wps.get(o, set())
            if not image_wps_o:
                return self.UNREACHABLE_COST # No waypoint to view objective o

            min_total_cost = float('inf')

            for r, i in capable_ri_pairs:
                r_loc = rover_locs.get(r)
                if r_loc is None: continue # Rover location unknown

                # Cost to calibrate (if needed)
                cost_calibrate_phase = 0
                loc_after_calib = r_loc
                is_calibrated = (i, r) in calibrated_cams
                calib_target = self.calib_targets.get(i)
                calib_wps_i = self.objective_calib_wps.get(i, set()) if calib_target else set() # Calib wps mapped by camera

                if not is_calibrated:
                    if not calib_wps_i:
                        # Cannot calibrate this camera, this (r, i) pair cannot achieve the goal
                        continue

                    min_calib_move = float('inf')
                    best_calib_wp = None
                    for w in calib_wps_i:
                        d = self._get_dist(r_loc, w)
                        if d < min_calib_move:
                            min_calib_move = d
                            best_calib_wp = w

                    if best_calib_wp is None or min_calib_move == float('inf'):
                         # Cannot reach any calibration waypoint
                         continue

                    cost_calibrate_phase = min_calib_move + 1 # +1 calibrate action
                    loc_after_calib = best_calib_wp

                # Cost to take image
                min_image_move = float('inf')
                best_image_wp = None
                for p in image_wps_o:
                    d = self._get_dist(loc_after_calib, p)
                    if d < min_image_move:
                        min_image_move = d
                        best_image_wp = p

                if best_image_wp is None or min_image_move == float('inf'):
                    # Cannot reach any image waypoint from location after calibration
                    continue

                cost_image_phase = min_image_move + 1 # +1 take_image action
                loc_after_image = best_image_wp

                # Cost to communicate after taking image (rover is at best_image_wp)
                comm_move_cost = self._min_dist_to_comm_wp(loc_after_image)
                if comm_move_cost == float('inf'): continue # Cannot reach communication waypoint from image waypoint

                total_cost = cost_calibrate_phase + cost_image_phase + comm_move_cost + 1 # +1 communicate action
                min_total_cost = min(min_total_cost, total_cost)

            return min_total_cost if min_total_cost != float('inf') else self.UNREACHABLE_COST


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse current state ---
        current_rover_locations = {}
        current_store_status = {}
        current_soil_samples = set()
        current_rock_samples = set()
        current_have_soil = collections.defaultdict(set)
        current_have_rock = collections.defaultdict(set)
        current_calibrated_cameras = set()
        current_have_image = set()
        current_communicated_soil = set()
        current_communicated_rock = set()
        current_communicated_image = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if match(fact, "at", "*", "*"):
                # Assuming anything 'at' a waypoint is a rover for location tracking
                obj, loc = parts[1], parts[2]
                current_rover_locations[obj] = loc
            elif match(fact, "empty", "*"):
                current_store_status[parts[1]] = 'empty'
            elif match(fact, "full", "*"):
                current_store_status[parts[1]] = 'full'
            elif match(fact, "at_soil_sample", "*"):
                current_soil_samples.add(parts[1])
            elif match(fact, "at_rock_sample", "*"):
                current_rock_samples.add(parts[1])
            elif match(fact, "have_soil_analysis", "*", "*"):
                rover, waypoint = parts[1], parts[2]
                current_have_soil[rover].add(waypoint)
            elif match(fact, "have_rock_analysis", "*", "*"):
                rover, waypoint = parts[1], parts[2]
                current_have_rock[rover].add(waypoint)
            elif match(fact, "calibrated", "*", "*"):
                camera, rover = parts[1], parts[2]
                current_calibrated_cameras.add((camera, rover))
            elif match(fact, "have_image", "*", "*", "*"):
                rover, objective, mode = parts[1], parts[2], parts[3]
                current_have_image.add((rover, objective, mode))
            elif match(fact, "communicated_soil_data", "*"):
                current_communicated_soil.add(parts[1])
            elif match(fact, "communicated_rock_data", "*"):
                current_communicated_rock.add(parts[1])
            elif match(fact, "communicated_image_data", "*", "*"):
                objective, mode = parts[1], parts[2]
                current_communicated_image.add((objective, mode))

        # --- Compute heuristic value ---
        h = 0

        for goal in self.goals:
            pred, *args = get_parts(goal)
            if not pred: continue # Skip invalid goals

            if pred == 'communicated_soil_data':
                w = args[0]
                if w not in current_communicated_soil:
                    h += self._cost_to_communicate_soil(w, state, current_rover_locations, current_store_status, current_have_soil)
            elif pred == 'communicated_rock_data':
                w = args[0]
                if w not in current_communicated_rock:
                     h += self._cost_to_communicate_rock(w, state, current_rover_locations, current_store_status, current_have_rock)
            elif pred == 'communicated_image_data':
                o, m = args
                if (o, m) not in current_communicated_image:
                     h += self._cost_to_communicate_image(o, m, state, current_rover_locations, current_calibrated_cameras, current_have_image)
            # Add other goal types if necessary, but these are the only ones in the domain file

        # Ensure heuristic is 0 for goal state
        if h == 0 and not self.goals <= state:
             # This should ideally not happen if costs are calculated correctly,
             # but as a safeguard, if the sum is 0 but goal is not reached,
             # it implies an issue or a state where 0 actions suffice (which is the goal state).
             # If the goal is not reached, the heuristic should be > 0.
             # The cost calculation logic should ensure this by returning > 0 for unachieved goals.
             # If all _cost_to_communicate methods return 0 for unachieved goals, there's a bug.
             # Let's rely on the cost calculation logic.
             pass # No adjustment needed if logic is correct

        return int(h) # Return integer heuristic value
