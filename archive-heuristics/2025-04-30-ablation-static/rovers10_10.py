from fnmatch import fnmatch
import collections
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper functions

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
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
         return False # Pattern length must match fact parts length
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """Computes shortest path distances from start_node in a graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
        return distances # Start node not in graph
    distances[start_node] = 0
    queue = collections.deque([start_node])
    while queue:
        current_node = queue.popleft()
        # Check if current_node is in graph before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# Define the heuristic class
class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all uncommunicated goals.
    It calculates the estimated cost for each unachieved goal independently, considering
    the steps needed for data acquisition (sampling or imaging) and communication.
    For each goal, it finds the minimum estimated cost among all capable rovers/cameras.
    Navigation costs are estimated using precomputed shortest paths on the traversal graph
    for each rover.

    # Assumptions
    - The heuristic ignores negative interactions between actions (e.g., camera uncalibration
      after taking an image for a different goal, or store becoming full/empty).
    - Resource contention (multiple goals needing the same rover/store/camera) is ignored;
      each goal's cost is calculated as if it had dedicated resources.
    - The cost of each PDDL action is 1.
    - Shortest paths are precomputed using BFS on the `can_traverse` graph, considering
      `visible` constraints for navigation.
    - A goal requiring soil/rock data from a waypoint where the sample is no longer present
      and no rover currently holds the data is considered unreachable by sampling.

    # Heuristic Initialization
    The constructor precomputes and stores static information from the task:
    - Lander location and waypoints visible from the lander (communication points).
    - Rover capabilities (equipped for soil, rock, imaging).
    - Rover store ownership.
    - Camera information (on which rover, supported modes, calibration target).
    - Objective information (waypoints visible from the objective).
    - Rover traversal graphs (based on `can_traverse` and `visible`).
    - All-pairs shortest paths for each rover on its traversal graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1.  Initialize total estimated cost to 0.
    2.  Identify all goal predicates that are not currently true in the state.
    3.  For each unachieved goal:
        a.  Determine the type of goal (soil data, rock data, or image data).
        b.  Find all rovers (and cameras, for image goals) that are capable of achieving this goal type based on their equipment and onboard cameras/modes.
        c.  If no capable resource exists, or if required locations (sample, cal, image, comm) are unreachable by any capable resource, the goal is considered unreachable, and a large penalty is added to the total cost.
        d.  If capable resources exist and locations are reachable, calculate the estimated cost for achieving this specific goal using *each* capable resource independently. This cost includes:
            i.  Cost for data acquisition (if data is not already held by any rover):
                -   1 action for sampling (soil/rock) or taking image.
                -   1 action for dropping sample if the rover's store is full (for soil/rock).
                -   1 action for calibrating camera if needed (for image).
                -   Navigation cost: Shortest path distance from the rover's current location to the required location(s) (sample waypoint, calibration waypoint, image waypoint). The location is updated after each required move (e.g., after calibration move, the rover is assumed to be at the calibration waypoint for the image move calculation).
            ii. Cost for communication (if data is not already communicated - this is always true for unmet goals):
                -   1 action for communicating data.
                -   Navigation cost: Shortest path distance from the rover's location (after acquisition) to the closest waypoint visible from the lander.
        e.  Take the minimum estimated cost among all capable resources for this goal.
        f.  Add this minimum cost to the total estimated cost.
    4.  Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.lander_location = None
        self.visible_map = collections.defaultdict(set) # wp -> {visible_wps}
        self.rover_capabilities = collections.defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_stores = {} # rover -> store
        self.camera_info = {} # camera -> {'rover': r, 'supports': {modes}, 'cal_target': t}
        self.objective_info = collections.defaultdict(lambda: {'visible_from': set()}) # objective -> {'visible_from': {wps}}
        self.rover_can_traverse = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> {wp -> {neighbors}}
        self.all_waypoints = set() # Collect all waypoints

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            pred = parts[0]
            if pred == 'at_lander' and len(parts) == 3:
                self.lander_location = parts[2]
            elif pred == 'visible' and len(parts) == 3:
                w1, w2 = parts[1], parts[2]
                self.visible_map[w1].add(w2)
                self.visible_map[w2].add(w1)
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
            elif pred == 'can_traverse' and len(parts) == 4:
                r, w1, w2 = parts[1], parts[2], parts[3]
                self.rover_can_traverse[r][w1].add(w2)
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
            elif pred == 'equipped_for_soil_analysis' and len(parts) == 2:
                self.rover_capabilities[parts[1]].add('soil')
            elif pred == 'equipped_for_rock_analysis' and len(parts) == 2:
                self.rover_capabilities[parts[1]].add('rock')
            elif pred == 'equipped_for_imaging' and len(parts) == 2:
                self.rover_capabilities[parts[1]].add('imaging')
            elif pred == 'store_of' and len(parts) == 3:
                self.rover_stores[parts[2]] = parts[1] # rover -> store
            elif pred == 'supports' and len(parts) == 3:
                c, m = parts[1], parts[2]
                self.camera_info.setdefault(c, {}).setdefault('supports', set()).add(m)
            elif pred == 'on_board' and len(parts) == 3:
                i, r = parts[1], parts[2]
                self.camera_info.setdefault(i, {})['rover'] = r
            elif pred == 'calibration_target' and len(parts) == 3:
                i, t = parts[1], parts[2]
                self.camera_info.setdefault(i, {})['cal_target'] = t
            elif pred == 'visible_from' and len(parts) == 3:
                o, w = parts[1], parts[2]
                self.objective_info.setdefault(o, {}).setdefault('visible_from', set()).add(w)
                self.all_waypoints.add(w)

        # Compute communication waypoints visible from the lander
        self.lander_comm_wps = self.visible_map.get(self.lander_location, set())

        # Build rover traversal graphs and compute shortest paths
        self.rover_traversal_graph = collections.defaultdict(lambda: collections.defaultdict(list))
        self.rover_shortest_paths = {}

        for rover in self.rover_can_traverse: # Iterate only rovers that have can_traverse facts
             # Initialize graph nodes for this rover with all known waypoints
            graph_for_rover = {wp: [] for wp in self.all_waypoints}
            for u, neighbors in self.rover_can_traverse[rover].items():
                 if u in graph_for_rover: # Ensure waypoint is in the set of all waypoints
                    for v in neighbors:
                        # Edge exists if can_traverse AND visible
                        if v in self.visible_map.get(u, set()):
                             if v in graph_for_rover: # Ensure neighbor waypoint is valid
                                graph_for_rover[u].append(v)
            self.rover_traversal_graph[rover] = graph_for_rover

            # Compute shortest paths for this rover
            self.rover_shortest_paths[rover] = {}
            for start_wp in self.all_waypoints:
                self.rover_shortest_paths[rover][start_wp] = bfs(self.rover_traversal_graph.get(rover, {}), start_wp) # Handle rover with empty graph


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract relevant current state facts
        rover_locations = {}
        store_states = {} # store -> 'full' or 'empty'
        have_soil = set() # (rover, waypoint)
        have_rock = set() # (rover, waypoint)
        have_image = set() # (rover, objective, mode)
        calibrated = set() # (camera, rover)
        at_soil_sample = set() # waypoint
        at_rock_sample = set() # waypoint

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at' and len(parts) == 3:
                 # Check if the second part is a rover type (heuristic assumption)
                 # A more robust check would involve knowing object types
                 # For simplicity, assume 'at roverX waypointY' format
                 if parts[1].startswith('rover'): # Simple check for rover type
                    rover_locations[parts[1]] = parts[2]
            elif pred == 'full' and len(parts) == 2:
                store_states[parts[1]] = 'full'
            elif pred == 'empty' and len(parts) == 2:
                 store_states[parts[1]] = 'empty'
            elif pred == 'have_soil_analysis' and len(parts) == 3:
                have_soil.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis' and len(parts) == 3:
                have_rock.add((parts[1], parts[2]))
            elif pred == 'have_image' and len(parts) == 4:
                have_image.add((parts[1], parts[2], parts[3]))
            elif pred == 'calibrated' and len(parts) == 3:
                calibrated.add((parts[1], parts[2]))
            elif pred == 'at_soil_sample' and len(parts) == 2:
                 at_soil_sample.add(parts[1])
            elif pred == 'at_rock_sample' and len(parts) == 2:
                 at_rock_sample.add(parts[1])


        total_cost = 0
        unmet_goals = self.goals - state

        # Penalty for unreachable goals
        UNREACHABLE_PENALTY = 1000

        for goal in unmet_goals:
            parts = get_parts(goal)
            if not parts: continue

            pred = parts[0]
            min_goal_cost = float('inf')

            if pred == 'communicated_soil_data' and len(parts) == 2:
                w_sample = parts[1]
                # Check if data is already acquired by ANY rover
                data_acquired = any((r, w_sample) in have_soil for r in self.rover_capabilities if 'soil' in self.rover_capabilities[r])

                # If data is not acquired, check if sampling is possible (sample exists)
                if not data_acquired and w_sample not in at_soil_sample:
                     # Sample is gone and no rover has the data - unreachable via sampling
                     min_goal_cost = float('inf')
                else:
                    # Find capable rovers
                    capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]
                    if not capable_rovers:
                        min_goal_cost = float('inf')
                    else:
                        for rover in capable_rovers:
                            current_rover_cost = 0
                            loc_r = rover_locations.get(rover)
                            if loc_r is None: continue # Rover location unknown or not 'at' any waypoint

                            loc_r_after_acquire = loc_r # Default location

                            if not data_acquired:
                                # Cost to acquire data (sample)
                                current_rover_cost += 1 # sample_soil action
                                store = self.rover_stores.get(rover)
                                # Check if store exists and is full
                                if store and store_states.get(store) == 'full':
                                    current_rover_cost += 1 # drop action

                                # Navigation to sample location
                                dist_to_sample = self.rover_shortest_paths.get(rover, {}).get(loc_r, {}).get(w_sample, float('inf'))
                                if dist_to_sample == float('inf'): continue # Cannot reach sample location
                                current_rover_cost += dist_to_sample
                                loc_r_after_acquire = w_sample # Rover is at sample location

                            # Cost to communicate data
                            current_rover_cost += 1 # communicate_soil_data action
                            min_comm_dist = float('inf')
                            if self.lander_comm_wps:
                                # Find closest comm waypoint from location after acquisition
                                min_comm_dist = min(self.rover_shortest_paths.get(rover, {}).get(loc_r_after_acquire, {}).get(w, float('inf')) for w in self.lander_comm_wps)

                            if min_comm_dist == float('inf'): continue # Cannot reach any comm location
                            current_rover_cost += min_comm_dist

                            min_goal_cost = min(min_goal_cost, current_rover_cost)

            elif pred == 'communicated_rock_data' and len(parts) == 2:
                w_sample = parts[1]
                # Check if data is already acquired by ANY rover
                data_acquired = any((r, w_sample) in have_rock for r in self.rover_capabilities if 'rock' in self.rover_capabilities[r])

                # If data is not acquired, check if sampling is possible (sample exists)
                if not data_acquired and w_sample not in at_rock_sample:
                     min_goal_cost = float('inf') # Cannot sample if sample is gone
                else:
                    # Find capable rovers
                    capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]
                    if not capable_rovers:
                        min_goal_cost = float('inf')
                    else:
                        for rover in capable_rovers:
                            current_rover_cost = 0
                            loc_r = rover_locations.get(rover)
                            if loc_r is None: continue # Rover location unknown

                            loc_r_after_acquire = loc_r # Default location

                            if not data_acquired:
                                # Cost to acquire data (sample)
                                current_rover_cost += 1 # sample_rock action
                                store = self.rover_stores.get(rover)
                                # Check if store exists and is full
                                if store and store_states.get(store) == 'full':
                                    current_rover_cost += 1 # drop action

                                # Navigation to sample location
                                dist_to_sample = self.rover_shortest_paths.get(rover, {}).get(loc_r, {}).get(w_sample, float('inf'))
                                if dist_to_sample == float('inf'): continue # Cannot reach sample location
                                current_rover_cost += dist_to_sample
                                loc_r_after_acquire = w_sample # Rover is at sample location

                            # Cost to communicate data
                            current_rover_cost += 1 # communicate_rock_data action
                            min_comm_dist = float('inf')
                            if self.lander_comm_wps:
                                # Find closest comm waypoint from location after acquisition
                                min_comm_dist = min(self.rover_shortest_paths.get(rover, {}).get(loc_r_after_acquire, {}).get(w, float('inf')) for w in self.lander_comm_wps)

                            if min_comm_dist == float('inf'): continue # Cannot reach any comm location
                            current_rover_cost += min_comm_dist

                            min_goal_cost = min(min_goal_cost, current_rover_cost)

            elif pred == 'communicated_image_data' and len(parts) == 3:
                o, m = parts[1], parts[2]
                # Find capable pairs (rover, camera)
                capable_pairs = [(r, c) for r, caps in self.rover_capabilities.items() if 'imaging' in caps
                                 for c, info in self.camera_info.items() if info.get('rover') == r and m in info.get('supports', set())]

                if not capable_pairs:
                    min_goal_cost = float('inf')
                else:
                    # Check if data is already acquired by ANY capable pair
                    data_acquired = any((r, o, m) in have_image for r, c in capable_pairs)

                    for rover, camera in capable_pairs:
                        current_pair_cost = 0
                        loc_r = rover_locations.get(rover)
                        if loc_r is None: continue # Rover location unknown

                        loc_r_after_acquire = loc_r # Default location

                        if not data_acquired:
                            # Cost to acquire data (take image)
                            current_pair_cost += 1 # take_image action
                            loc_r_after_cal = loc_r # Default location after calibration

                            if (camera, rover) not in calibrated:
                                # Cost to calibrate
                                current_pair_cost += 1 # calibrate action
                                cal_target = self.camera_info.get(camera, {}).get('cal_target')
                                if not cal_target: continue # Camera has no calibration target

                                cal_wps = self.objective_info.get(cal_target, {}).get('visible_from', set())
                                if not cal_wps: continue # No calibration waypoints for target

                                # Navigation to calibration location
                                min_cal_dist = float('inf')
                                closest_cal_wp = None
                                if cal_wps:
                                     min_dist_finder = float('inf')
                                     for w in cal_wps:
                                          dist = self.rover_shortest_paths.get(rover, {}).get(loc_r, {}).get(w, float('inf'))
                                          if dist < min_dist_finder:
                                               min_dist_finder = dist
                                               closest_cal_wp = w
                                     min_cal_dist = min_dist_finder

                                if min_cal_dist == float('inf'): continue # Cannot reach any cal location
                                current_pair_cost += min_cal_dist
                                loc_r_after_cal = closest_cal_wp # Rover is at cal location after calibrating

                            # Navigation to image location
                            img_wps = self.objective_info.get(o, {}).get('visible_from', set())
                            if not img_wps: continue # No image waypoints for objective

                            min_img_dist = float('inf')
                            closest_img_wp = None
                            if img_wps:
                                min_dist_finder = float('inf')
                                for w in img_wps:
                                    dist = self.rover_shortest_paths.get(rover, {}).get(loc_r_after_cal, {}).get(w, float('inf'))
                                    if dist < min_dist_finder:
                                         min_dist_finder = dist
                                         closest_img_wp = w
                                min_img_dist = min_dist_finder

                            if min_img_dist == float('inf'): continue # Cannot reach any image location
                            current_pair_cost += min_img_dist
                            loc_r_after_acquire = closest_img_wp # Rover is at image location after imaging

                        # Cost to communicate data
                        current_pair_cost += 1 # communicate_image_data action
                        min_comm_dist = float('inf')
                        if self.lander_comm_wps:
                            # Find closest comm waypoint from location after acquisition
                            min_comm_dist = min(self.rover_shortest_paths.get(rover, {}).get(loc_r_after_acquire, {}).get(w, float('inf')) for w in self.lander_comm_wps)

                        if min_comm_dist == float('inf'): continue # Cannot reach any comm location
                        current_pair_cost += min_comm_dist

                        min_goal_cost = min(min_goal_cost, current_pair_cost)

            # Add cost for this goal to total cost
            if min_goal_cost == float('inf'):
                total_cost += UNREACHABLE_PENALTY
            else:
                total_cost += min_goal_cost

        return total_cost
