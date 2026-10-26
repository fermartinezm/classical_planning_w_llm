from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """Performs Breadth-First Search to find shortest distances from start_node."""
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = collections.deque([start_node])
    while queue:
        current_node = queue.popleft()
        for neighbor in graph.get(current_node, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest path distances between all pairs of nodes in the graph."""
    all_dist = {}
    for node in graph:
        all_dist[node] = bfs(graph, node)
    return all_dist


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions, focusing on the communication goals (soil, rock, image data).
    It considers the steps needed to acquire the data (sampling, imaging,
    calibrating) and the navigation costs between relevant locations (sample/image
    waypoints, calibration waypoints, communication waypoints). It sums the
    estimated costs for each uncommunicated goal independently.

    # Assumptions
    - Each action (navigate, sample, drop, calibrate, take_image, communicate) costs 1.
    - Navigation cost between two waypoints is the shortest path distance in the
      waypoint graph defined by the `visible` predicate.
    - The heuristic sums costs for achieving each goal fact independently, ignoring
      potential negative interactions (beyond a simplified check for needing a store drop).
    - When data acquisition is needed, the heuristic estimates the minimum navigation
      cost over all suitable rovers, cameras, and waypoints.
    - When communication is needed, the heuristic estimates the minimum navigation
      cost from the current data location (or acquisition location) to any
      waypoint visible from the lander.
    - Problems are assumed to be solvable if the waypoint graph is connected
      and required samples/visibility exist. Unreachable required locations result
      in an infinite heuristic value.

    # Heuristic Initialization
    The constructor precomputes static information from the task:
    - Builds the waypoint graph based on `visible` predicates.
    - Computes all-pairs shortest path distances between waypoints using BFS.
    - Identifies the lander location and the set of waypoints visible from the lander.
    - Stores which rovers are equipped for soil, rock, and imaging analysis.
    - Stores the mapping between rovers and their stores.
    - Stores camera information: which rover they are on, which modes they support,
      and their calibration targets.
    - Stores objective visibility information: which waypoints an objective is
      visible from.
    - Extracts the set of goal conditions, categorizing them into required
      communicated soil data (by waypoint), rock data (by waypoint), and image
      data (by objective and mode).
    - Extracts initial locations of soil and rock samples.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the estimated cost as follows:

    1.  Initialize the total heuristic cost `h` to 0.
    2.  Extract the current state information: rover locations, which rovers have
        which data (soil, rock, image), which stores are full, which cameras are
        calibrated, and which data has already been communicated.
    3.  For each waypoint `w` for which `(communicated_soil_data w)` is a goal
        and is not true in the current state:
        a.  Add 1 to `h` for the `communicate_soil_data` action.
        b.  Check if any rover currently has `(have_soil_analysis r w)`.
        c.  If no rover has the data:
            i.  Add 1 to `h` for the `sample_soil` action.
            ii. Check if any equipped soil rover has an empty store. If all equipped
                soil rovers have full stores, add 1 to `h` for a `drop` action.
            iii. Calculate the minimum navigation cost from any suitable equipped
                 soil rover's current location to waypoint `w`. Add this cost to `h`.
            iv. Calculate the minimum navigation cost from waypoint `w` (where sampling
                occurs) to any waypoint visible from the lander. Add this cost to `h`.
        d.  If a rover `r_h` has the data:
            i.  Calculate the minimum navigation cost from `r_h`'s current location
                to any waypoint visible from the lander. Add this cost to `h`.
    4.  For each waypoint `w` for which `(communicated_rock_data w)` is a goal
        and is not true in the current state: (Follows similar logic as soil data,
        using rock-specific predicates and equipment).
    5.  For each objective-mode pair `(o, m)` for which `(communicated_image_data o m)`
        is a goal and is not true in the current state:
        a.  Add 1 to `h` for the `communicate_image_data` action.
        b.  Check if any rover currently has `(have_image r o m)`.
        c.  If no rover has the data:
            i.  Add 1 to `h` for the `take_image` action.
            ii. Calculate the minimum cost to acquire the image data, considering
                all suitable rovers, cameras, calibration waypoints, and image waypoints.
                This cost includes:
                -   If the chosen camera needs calibration: 1 for `calibrate` action
                    plus navigation from the rover's current location to a calibration
                    waypoint for that camera's target, then to an image waypoint
                    for the objective `o`.
                -   If the chosen camera is already calibrated: Navigation from the
                    rover's current location directly to an image waypoint for `o`.
                Add this minimum acquisition cost to `h`.
            iii. Calculate the minimum navigation cost from any image waypoint for
                 objective `o` to any waypoint visible from the lander. Add this cost to `h`.
        d.  If a rover `r_h` has the data:
            i.  Calculate the minimum navigation cost from `r_h`'s current location
                to any waypoint visible from the lander. Add this cost to `h`.
    6.  If at any point a required navigation path is impossible (waypoints are
        disconnected), return `float('inf')`.
    7.  Return the total calculated cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Needed to check initial sample locations

        # --- Precompute Waypoint Graph and Distances ---
        self.graph = collections.defaultdict(set)
        waypoints = set()
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                _, w1, w2 = get_parts(fact)
                self.graph[w1].add(w2)
                self.graph[w2].add(w1) # Assume visibility is symmetric
                waypoints.add(w1)
                waypoints.add(w2)

        # Ensure all waypoints mentioned in other static facts are in the graph nodes
        for fact in static_facts:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith('waypoint'):
                     waypoints.add(part)
        for wp in waypoints:
             if wp not in self.graph:
                 self.graph[wp] = set() # Add isolated waypoints

        self.dist = compute_all_pairs_shortest_paths(self.graph)

        # --- Extract Static Information ---
        self.lander_wp = None
        self.comm_wps = set() # Waypoints visible from the lander
        self.equipped_soil_rovers = set()
        self.equipped_rock_rovers = set()
        self.equipped_imaging_rovers = set()
        self.rover_to_store = {}
        self.camera_on_board = collections.defaultdict(set) # rover -> {cameras}
        self.camera_supports = collections.defaultdict(set) # camera -> {modes}
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visible_from = collections.defaultdict(set) # objective -> {waypoints}

        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at_lander":
                self.lander_wp = parts[2]
            elif predicate == "equipped_for_soil_analysis":
                self.equipped_soil_rovers.add(parts[1])
            elif predicate == "equipped_for_rock_analysis":
                self.equipped_rock_rovers.add(parts[1])
            elif predicate == "equipped_for_imaging":
                self.equipped_imaging_rovers.add(parts[1])
            elif predicate == "store_of":
                store, rover = parts[1], parts[2]
                self.rover_to_store[rover] = store # Assuming one store per rover
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.camera_on_board[rover].add(camera)
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_supports[camera].add(mode)
            elif predicate == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.camera_calibration_target[camera] = objective
            elif predicate == "visible_from":
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from[objective].add(waypoint)

        # Identify communication waypoints (visible from lander)
        if self.lander_wp and self.lander_wp in self.graph:
             self.comm_wps = set(self.graph[self.lander_wp])


        # --- Extract Goal Information ---
        self.goal_soil_wps = set()
        self.goal_rock_wps = set()
        self.goal_image_tasks = set() # Stores (objective, mode) tuples

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            predicate = parts[0]
            if predicate == "communicated_soil_data":
                self.goal_soil_wps.add(parts[1])
            elif predicate == "communicated_rock_data":
                self.goal_rock_wps.add(parts[1])
            elif predicate == "communicated_image_data":
                self.goal_image_tasks.add((parts[1], parts[2]))

        # --- Extract Initial Sample Information ---
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        for fact in initial_state:
            parts = get_parts(fact)
            if parts[0] == "at_soil_sample":
                self.initial_soil_samples.add(parts[1])
            elif parts[0] == "at_rock_sample":
                self.initial_rock_samples.add(parts[1])


    def min_dist_to_set(self, start_wp, target_wp_set):
        """Returns min distance from start_wp to any waypoint in target_wp_set."""
        if start_wp not in self.dist:
            return float('inf') # Start waypoint is not in the graph
        min_d = float('inf')
        for target_wp in target_wp_set:
            if target_wp in self.dist[start_wp]:
                min_d = min(min_d, self.dist[start_wp][target_wp])
        return min_d

    def min_dist_to_set_from_set(self, start_wp_set, target_wp_set):
        """Returns min distance from any waypoint in start_wp_set to any waypoint in target_wp_set."""
        min_d = float('inf')
        for start_wp in start_wp_set:
            min_d = min(min_d, self.min_dist_to_set(start_wp, target_wp_set))
        return min_d


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # --- Extract Current State Information ---
        rover_locs = {} # rover -> waypoint
        current_have_soil = {} # (rover, wp) -> True
        current_have_rock = {} # (rover, wp) -> True
        current_have_image = {} # (rover, obj, mode) -> True
        current_store_full = {} # store -> True
        current_calibrated_cam = {} # (camera, rover) -> True
        current_comm_soil = {} # wp -> True
        current_comm_rock = {} # wp -> True
        current_comm_image = {} # (obj, mode) -> True

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at" and parts[1].startswith('rover'):
                rover_locs[parts[1]] = parts[2]
            elif predicate == "have_soil_analysis":
                current_have_soil[(parts[1], parts[2])] = True
            elif predicate == "have_rock_analysis":
                current_have_rock[(parts[1], parts[2])] = True
            elif predicate == "have_image":
                current_have_image[(parts[1], parts[2], parts[3])] = True
            elif predicate == "full":
                current_store_full[parts[1]] = True
            elif predicate == "calibrated":
                current_calibrated_cam[(parts[1], parts[2])] = True
            elif predicate == "communicated_soil_data":
                current_comm_soil[parts[1]] = True
            elif predicate == "communicated_rock_data":
                current_comm_rock[parts[1]] = True
            elif predicate == "communicated_image_data":
                current_comm_image[(parts[1], parts[2])] = True

        # --- Estimate Cost for Uncommunicated Goals ---

        # Soil Data Goals
        for soil_wp in self.goal_soil_wps:
            if not current_comm_soil.get(soil_wp, False):
                h += 1 # communicate_soil_data action needed

                # Check if data is already collected by any equipped rover
                has_data = any((r, soil_wp) in current_have_soil for r in self.equipped_soil_rovers)

                if not has_data:
                    # Need to sample
                    h += 1 # sample_soil action needed

                    # Check store constraint for equipped rovers
                    needs_drop = True # Assume drop is needed unless an empty store is found
                    suitable_rovers_for_sampling = []
                    for r in self.equipped_soil_rovers:
                        store = self.rover_to_store.get(r)
                        if store and not current_store_full.get(store, False):
                            needs_drop = False
                            suitable_rovers_for_sampling.append(r)

                    if needs_drop:
                         h += 1 # drop action needed by some rover

                    # Navigation to sample location (consider rovers that can potentially sample)
                    rovers_to_consider = self.equipped_soil_rovers if needs_drop else suitable_rovers_for_sampling
                    min_nav_to_sample = float('inf')
                    for r in rovers_to_consider:
                        current_loc = rover_locs.get(r)
                        if current_loc and current_loc in self.dist and soil_wp in self.dist[current_loc]:
                             min_nav_to_sample = min(min_nav_to_sample, self.dist[current_loc][soil_wp])

                    if min_nav_to_sample == float('inf'): return float('inf') # Unreachable sample
                    h += min_nav_to_sample

                    # Navigation from sample location to communication location
                    min_nav_to_comm = self.min_dist_to_set(soil_wp, self.comm_wps)
                    if min_nav_to_comm == float('inf'): return float('inf') # Unreachable communication point
                    h += min_nav_to_comm

                else: # Data is already collected
                    # Navigation from current data location to communication location
                    min_nav_to_comm = float('inf')
                    for r in self.equipped_soil_rovers: # Check all equipped rovers, one might have the data
                        if (r, soil_wp) in current_have_soil:
                            current_loc = rover_locs.get(r)
                            if current_loc and current_loc in self.dist:
                                min_nav_to_comm = min(min_nav_to_comm, self.min_dist_to_set(current_loc, self.comm_wps))

                    if min_nav_to_comm == float('inf'): return float('inf') # Rover with data cannot reach communication point
                    h += min_nav_to_comm


        # Rock Data Goals
        for rock_wp in self.goal_rock_wps:
            if not current_comm_rock.get(rock_wp, False):
                h += 1 # communicate_rock_data action needed

                # Check if data is already collected by any equipped rover
                has_data = any((r, rock_wp) in current_have_rock for r in self.equipped_rock_rovers)

                if not has_data:
                    # Need to sample
                    h += 1 # sample_rock action needed

                    # Check store constraint for equipped rovers
                    needs_drop = True
                    suitable_rovers_for_sampling = []
                    for r in self.equipped_rock_rovers:
                        store = self.rover_to_store.get(r)
                        if store and not current_store_full.get(store, False):
                            needs_drop = False
                            suitable_rovers_for_sampling.append(r)

                    if needs_drop:
                         h += 1 # drop action needed

                    # Navigation to sample location
                    rovers_to_consider = self.equipped_rock_rovers if needs_drop else suitable_rovers_for_sampling
                    min_nav_to_sample = float('inf')
                    for r in rovers_to_consider:
                        current_loc = rover_locs.get(r)
                        if current_loc and current_loc in self.dist and rock_wp in self.dist[current_loc]:
                             min_nav_to_sample = min(min_nav_to_sample, self.dist[current_loc][rock_wp])

                    if min_nav_to_sample == float('inf'): return float('inf') # Unreachable sample
                    h += min_nav_to_sample

                    # Navigation from sample location to communication location
                    min_nav_to_comm = self.min_dist_to_set(rock_wp, self.comm_wps)
                    if min_nav_to_comm == float('inf'): return float('inf') # Unreachable communication point
                    h += min_nav_to_comm

                else: # Data is already collected
                    # Navigation from current data location to communication location
                    min_nav_to_comm = float('inf')
                    for r in self.equipped_rock_rovers: # Check all equipped rovers
                        if (r, rock_wp) in current_have_rock:
                            current_loc = rover_locs.get(r)
                            if current_loc and current_loc in self.dist:
                                min_nav_to_comm = min(min_nav_to_comm, self.min_dist_to_set(current_loc, self.comm_wps))

                    if min_nav_to_comm == float('inf'): return float('inf') # Rover with data cannot reach communication point
                    h += min_nav_to_comm


        # Image Data Goals
        for obj, mode in self.goal_image_tasks:
            if not current_comm_image.get((obj, mode), False):
                h += 1 # communicate_image_data action needed

                # Check if data is already collected by any equipped rover
                has_data = any((r, obj, mode) in current_have_image for r in self.equipped_imaging_rovers)

                if not has_data:
                    # Need to acquire image
                    h += 1 # take_image action needed

                    min_imaging_acquisition_cost = float('inf') # Includes calibrate action if needed + navigation

                    # Find best rover/camera/cal_wp/img_wp path
                    for r in self.equipped_imaging_rovers:
                        current_loc = rover_locs.get(r)
                        if current_loc is None or current_loc not in self.dist: continue # Rover location unknown or disconnected

                        for i in self.camera_on_board.get(r, set()):
                            if mode in self.camera_supports.get(i, set()):
                                # This is a suitable rover/camera pair (r, i) for mode

                                if (i, r) not in current_calibrated_cam: # Needs calibration
                                    cal_target = self.camera_calibration_target.get(i)
                                    if cal_target is None: continue # Camera has no calibration target
                                    cal_wps = self.objective_visible_from.get(cal_target, set())
                                    img_wps = self.objective_visible_from.get(obj, set())
                                    if not cal_wps or not img_wps: continue # No suitable waypoints

                                    # Find min path: current_loc -> cal_wp -> img_wp
                                    min_path_via_cal = float('inf')
                                    for cal_wp in cal_wps:
                                        for img_wp in img_wps:
                                            if cal_wp in self.dist.get(current_loc, {}) and img_wp in self.dist.get(cal_wp, {}):
                                                min_path_via_cal = min(min_path_via_cal, self.dist[current_loc][cal_wp] + self.dist[cal_wp][img_wp])

                                    if min_path_via_cal != float('inf'):
                                        # Cost is calibrate (1) + navigation
                                        min_imaging_acquisition_cost = min(min_imaging_acquisition_cost, 1 + min_path_via_cal)

                                else: # Already calibrated
                                    img_wps = self.objective_visible_from.get(obj, set())
                                    if not img_wps: continue # No suitable waypoints

                                    # Find min path: current_loc -> img_wp
                                    min_path_direct = float('inf')
                                    for img_wp in img_wps:
                                        if img_wp in self.dist.get(current_loc, {}):
                                            min_path_direct = min(min_path_direct, self.dist[current_loc][img_wp])

                                    if min_path_direct != float('inf'):
                                        # Cost is just navigation
                                        min_imaging_acquisition_cost = min(min_imaging_acquisition_cost, min_path_direct)

                    if min_imaging_acquisition_cost == float('inf'): return float('inf') # Unsolvable image acquisition
                    h += min_imaging_acquisition_cost

                    # Navigation from image location to communication location
                    # The rover is at *some* img_wp after acquisition.
                    # Find min distance from *any* image waypoint for this objective to *any* communication waypoint.
                    img_wps = self.objective_visible_from.get(obj, set())
                    min_nav_img_to_comm = self.min_dist_to_set_from_set(img_wps, self.comm_wps)
                    if min_nav_img_to_comm == float('inf'): return float('inf') # Unreachable communication point
                    h += min_nav_img_to_comm

                else: # Data is already collected
                    # Navigation from current data location to communication location
                    min_nav_to_comm = float('inf')
                    for r in self.equipped_imaging_rovers: # Check all equipped rovers
                        if (r, obj, mode) in current_have_image:
                            current_loc = rover_locs.get(r)
                            if current_loc and current_loc in self.dist:
                                min_nav_to_comm = min(min_nav_to_comm, self.min_dist_to_set(current_loc, self.comm_wps))

                    if min_nav_to_comm == float('inf'): return float('inf') # Rover with data cannot reach communication point
                    h += min_nav_to_comm


        return h
