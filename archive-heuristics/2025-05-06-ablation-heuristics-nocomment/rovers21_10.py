from heuristics.heuristic_base import Heuristic
import fnmatch
from collections import deque

# Helper functions
def get_parts(fact):
    """Helper to parse a fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match_fnmatch(fact, *args):
    """Helper to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """Computes shortest path distances from start_node in a graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node is not in the graph, cannot reach anything
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is a valid key in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the cost to reach a goal state by summing up the
    estimated costs for each unsatisfied goal predicate. The estimated cost
    for a single goal (e.g., communicated soil data from waypoint W) is the
    minimum cost over all capable rovers/cameras to first acquire the necessary
    data (sample or image) and then communicate it to the lander. Costs include
    navigation (estimated by shortest path on the can_traverse graph) and
    action costs (e.g., sample, drop, calibrate, take_image, communicate),
    each counted as 1 action.

    Assumptions:
    - There is exactly one lander.
    - Navigation cost between adjacent waypoints (connected by can_traverse) is 1.
    - The 'visible' precondition for navigation is implicitly handled by the
      can_traverse graph structure for shortest paths.
    - Action costs (sample, drop, calibrate, take_image, communicate) are 1.
    - Resource constraints (like store capacity or camera calibration state)
      are only partially modeled (e.g., adding drop cost if store is full *when needed*).
    - The heuristic sums costs for individual goals independently, ignoring
      potential synergies (e.g., one navigation action serving multiple goals)
      or conflicts (e.g., multiple goals requiring the same limited resource).
    - Unreachable goals or required intermediate steps result in a large finite cost (1000).

    Heuristic Initialization:
    During initialization, the heuristic precomputes static information from
    the task definition to speed up heuristic calculation during search:
    - Lander locations.
    - Rover capabilities (equipped for soil, rock, imaging).
    - Store ownership (mapping stores to rovers and vice versa).
    - Camera information (on which rover, supported modes, calibration target).
    - Waypoints where objectives/calibration targets are visible from.
    - Waypoints with initial soil/rock samples.
    - Waypoints visible from the lander location(s) (communication points).
    - For each rover, a navigation graph is built based on 'can_traverse' facts.
      All-pairs shortest paths are computed on this graph using BFS and stored.
    - Initial rover locations are stored.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state to extract dynamic information: current rover
        locations, store statuses (empty/full), data held by rovers (have_soil,
        have_rock, have_image), and camera calibration status.
    2.  Initialize the total heuristic value `h` to 0.
    3.  Iterate through each goal predicate defined in the task.
    4.  For each goal predicate, check if it is already satisfied in the current state.
    5.  If a goal predicate is *not* satisfied:
        - Estimate the minimum cost to satisfy this specific goal, considering
          all capable rovers and cameras.
        - This estimation involves two main phases:
            a.  Cost to acquire the necessary data (soil sample, rock sample, or image):
                - If the required 'have_X' predicate is not true for any rover:
                    - Find a rover capable of acquiring this data type.
                    - Calculate the shortest path cost from the rover's current
                      location to the required sampling/imaging waypoint.
                    - Add 1 for the sample/take_image action.
                    - For sampling, add 1 for a 'drop' action if the rover's
                      store is currently full.
                    - For imaging, if the required camera is not calibrated,
                      calculate the shortest path cost from the rover's current
                      location to a suitable calibration waypoint and add 1 for
                      the 'calibrate' action.
                    - The location after acquiring data is the sampling/imaging waypoint.
                - If the required 'have_X' predicate *is* true for some rover:
                    - The cost to acquire data is 0.
                    - The location after acquiring data is the rover's current location.
            b.  Cost to communicate the data:
                - Find a rover that holds the required data (either just acquired
                  or already held).
                - Calculate the shortest path cost from the rover's location
                  (after acquiring data, or current location if data was held)
                  to the nearest communication waypoint (visible from the lander).
                - Add 1 for the 'communicate_X_data' action.
        - The estimated cost for this goal is the minimum sum of costs (acquire + communicate)
          over all capable rovers/cameras.
        - If no capable rover/camera can achieve the goal (e.g., cannot reach
          required waypoints, no sample exists, no suitable camera), the cost
          is set to a large finite value (1000).
        - Add this estimated minimum cost to the total heuristic value `h`.
        - If at any point the total `h` reaches or exceeds 1000, return 1000
          immediately, indicating the state is likely unsolvable.
    6.  Return the total heuristic value `h`.
    """
    def __init__(self, task):
        super().__init__(task)

        # --- Parse Static Information ---
        self.lander_locations = {} # lander -> waypoint
        self.equipped_for_soil = set() # set of rovers
        self.equipped_for_rock = set() # set of rovers
        self.equipped_for_imaging = set() # set of rovers
        self.store_to_rover = {} # store -> rover
        self.rover_to_store = {} # rover -> store
        self.camera_on_rover = {} # camera -> rover
        self.camera_modes = {} # camera -> set of modes
        self.camera_cal_target = {} # camera -> objective
        self.objective_visible_wps = {} # objective -> set of waypoints
        self.caltarget_visible_wps = {} # objective (cal target) -> set of waypoints
        self.lander_comm_wps = {} # lander_wp -> set of visible waypoints
        self.initial_soil_samples = set() # set of waypoints
        self.initial_rock_samples = set() # set of waypoints
        self.initial_rover_locations = {} # rover -> waypoint

        all_waypoints = set() # Collect all waypoints mentioned anywhere

        # Parse initial state for initial rover locations
        for fact_str in self.initial_state:
             parts = get_parts(fact_str)
             if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('rover'):
                 self.initial_rover_locations[parts[1]] = parts[2]
                 all_waypoints.add(parts[2])
             elif parts[0] == 'at_lander' and len(parts) == 3:
                  all_waypoints.add(parts[2]) # Should be covered by static parsing too

        # Parse static facts
        for fact_str in self.static:
            parts = get_parts(fact_str)
            predicate = parts[0]
            if predicate == 'at_lander' and len(parts) == 3:
                lander, wp = parts[1], parts[2]
                self.lander_locations[lander] = wp
                all_waypoints.add(wp)
            elif predicate == 'equipped_for_soil_analysis' and len(parts) == 2:
                self.equipped_for_soil.add(parts[1])
            elif predicate == 'equipped_for_rock_analysis' and len(parts) == 2:
                self.equipped_for_rock.add(parts[1])
            elif predicate == 'equipped_for_imaging' and len(parts) == 2:
                self.equipped_for_imaging.add(parts[1])
            elif predicate == 'store_of' and len(parts) == 3:
                store, rover = parts[1], parts[2]
                self.store_to_rover[store] = rover
                self.rover_to_store[rover] = store
            elif predicate == 'on_board' and len(parts) == 3:
                camera, rover = parts[1], parts[2]
                self.camera_on_rover[camera] = rover
            elif predicate == 'supports' and len(parts) == 3:
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif predicate == 'calibration_target' and len(parts) == 3:
                camera, objective = parts[1], parts[2]
                self.camera_cal_target[camera] = objective
            elif predicate == 'visible_from' and len(parts) == 3:
                objective, wp = parts[1], parts[2]
                self.objective_visible_wps.setdefault(objective, set()).add(wp)
                all_waypoints.add(wp)
            elif predicate == 'at_soil_sample' and len(parts) == 2:
                self.initial_soil_samples.add(parts[1])
                all_waypoints.add(parts[1])
            elif predicate == 'at_rock_sample' and len(parts) == 2:
                self.initial_rock_samples.add(parts[1])
                all_waypoints.add(parts[1])
            elif predicate == 'visible' and len(parts) == 3:
                 wp1, wp2 = parts[1], parts[2]
                 all_waypoints.add(wp1)
                 all_waypoints.add(wp2)
                 # Process visible facts for lander communication points after lander locations are known
                 pass # Deferred processing

        # Process visible facts for lander communication points
        for fact_str in self.static:
            parts = get_parts(fact_str)
            if parts[0] == 'visible' and len(parts) == 3:
                wp1, wp2 = parts[1], parts[2]
                # Check if wp2 is a lander location
                for lander, lander_wp in self.lander_locations.items():
                    if wp2 == lander_wp:
                        self.lander_comm_wps.setdefault(lander_wp, set()).add(wp1)

        self.all_waypoints = list(all_waypoints) # Store all known waypoints

        # Build rover navigation graphs based on can_traverse
        self.rover_graphs = {} # rover -> {wp: [neighbors]}
        all_rovers_in_traverse = set()

        for fact_str in self.static:
            parts = get_parts(fact_str)
            if parts[0] == 'can_traverse' and len(parts) == 4:
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                self.rover_graphs.setdefault(rover, {}).setdefault(wp1, []).append(wp2)
                all_rovers_in_traverse.add(rover)

        # Ensure all rovers (from initial state or traverse) have graph entries
        all_rovers = set(self.initial_rover_locations.keys()).union(all_rovers_in_traverse)

        for rover in all_rovers:
            self.rover_graphs.setdefault(rover, {})
            # Ensure all known waypoints are keys in the graph for each rover
            for wp in self.all_waypoints:
                self.rover_graphs[rover].setdefault(wp, []) # Add waypoint as key if not present

        # Compute shortest paths for each rover from each waypoint
        self.rover_shortest_paths = {}
        for rover, graph in self.rover_graphs.items():
            self.rover_shortest_paths[rover] = {}
            for start_wp in graph: # Iterate through all waypoints in this rover's graph
                self.rover_shortest_paths[rover][start_wp] = bfs(graph, start_wp)

    def get_shortest_path_cost(self, rover, from_wp, to_wp):
        """Helper to get shortest path cost between two waypoints for a rover."""
        if rover not in self.rover_shortest_paths or \
           from_wp not in self.rover_shortest_paths[rover] or \
           to_wp not in self.rover_shortest_paths[rover][from_wp]:
            return float('inf') # Cannot find path
        return self.rover_shortest_paths[rover][from_wp][to_wp]

    def cost_soil_goal(self, state_info, w):
        """Estimates cost for a single communicated_soil_data goal."""
        # Check if sampling is even possible at this waypoint
        if w not in self.initial_soil_samples:
            return 1000 # Cannot sample soil at w, goal is impossible

        rover_locations = state_info['rover_locations']
        store_status = state_info['store_status']
        have_soil = state_info['have_soil']
        lander_wp = list(self.lander_locations.values())[0] # Assuming one lander
        comm_wps = self.lander_comm_wps.get(lander_wp, set())

        if not comm_wps: return 1000 # No communication points for lander

        min_total_cost = float('inf')

        # Iterate through all soil-equipped rovers
        for rover in self.equipped_for_soil:
            if rover not in rover_locations: continue # Rover not currently located

            current_wp = rover_locations[rover]
            rover_store = self.rover_to_store.get(rover)

            cost_to_get_data = 0
            # Check if this rover already has the data
            if w not in have_soil.get(rover, set()):
                # Need to sample
                nav_to_sample_cost = self.get_shortest_path_cost(rover, current_wp, w)
                if nav_to_sample_cost == float('inf'): continue # Cannot reach sample point

                cost_to_get_data = nav_to_sample_cost + 1 # navigate + sample

                # Add drop cost if store is full
                if rover_store and store_status.get(rover_store) == 'full':
                     cost_to_get_data += 1 # + drop cost

                # After sampling, rover is at waypoint w
                location_after_data = w
            else:
                # Rover already has data
                location_after_data = current_wp # Rover is still at its current location

            # Cost to communicate
            min_nav_to_comm_cost = float('inf')
            for comm_wp in comm_wps:
                nav_cost = self.get_shortest_path_cost(rover, location_after_data, comm_wp)
                min_nav_to_comm_cost = min(min_nav_to_comm_cost, nav_cost)

            if min_nav_to_comm_cost == float('inf'): continue # Cannot reach communication point

            comm_cost = min_nav_to_comm_cost + 1 # navigate + communicate

            total_cost_for_this_rover = cost_to_get_data + comm_cost
            min_total_cost = min(min_total_cost, total_cost_for_this_rover)

        return min_total_cost if min_total_cost != float('inf') else 1000 # Use a large finite value

    def cost_rock_goal(self, state_info, w):
        """Estimates cost for a single communicated_rock_data goal."""
        # Check if sampling is even possible at this waypoint
        if w not in self.initial_rock_samples:
            return 1000 # Cannot sample rock at w, goal is impossible

        rover_locations = state_info['rover_locations']
        store_status = state_info['store_status']
        have_rock = state_info['have_rock']
        lander_wp = list(self.lander_locations.values())[0] # Assuming one lander
        comm_wps = self.lander_comm_wps.get(lander_wp, set())

        if not comm_wps: return 1000

        min_total_cost = float('inf')

        for rover in self.equipped_for_rock:
            if rover not in rover_locations: continue

            current_wp = rover_locations[rover]
            rover_store = self.rover_to_store.get(rover)

            cost_to_get_data = 0
            if w not in have_rock.get(rover, set()):
                # Need to sample
                nav_to_sample_cost = self.get_shortest_path_cost(rover, current_wp, w)
                if nav_to_sample_cost == float('inf'): continue

                cost_to_get_data = nav_to_sample_cost + 1 # navigate + sample

                if rover_store and state_status.get(rover_store) == 'full':
                     cost_to_get_data += 1 # + drop cost

                location_after_data = w
            else:
                location_after_data = current_wp

            min_nav_to_comm_cost = float('inf')
            for comm_wp in comm_wps:
                nav_cost = self.get_shortest_path_cost(rover, location_after_data, comm_wp)
                min_nav_to_comm_cost = min(min_nav_to_comm_cost, nav_cost)

            if min_nav_to_comm_cost == float('inf'): continue

            comm_cost = min_nav_to_comm_cost + 1 # navigate + communicate

            total_cost_for_this_rover = cost_to_get_data + comm_cost
            min_total_cost = min(min_total_cost, total_cost_for_this_rover)

        return min_total_cost if min_total_cost != float('inf') else 1000

    def cost_image_goal(self, state_info, o, m):
        """Estimates cost for a single communicated_image_data goal."""
        rover_locations = state_info['rover_locations']
        have_image = state_info['have_image']
        camera_calibrated = state_info['camera_calibrated']
        lander_wp = list(self.lander_locations.values())[0] # Assuming one lander
        comm_wps = self.lander_comm_wps.get(lander_wp, set())
        img_wps = self.objective_visible_wps.get(o, set())

        if not comm_wps or not img_wps: return 1000 # Cannot communicate or image

        min_total_cost = float('inf')

        # Iterate through all imaging-equipped rovers with suitable cameras
        for rover in self.equipped_for_imaging:
            if rover not in rover_locations: continue

            current_wp = rover_locations[rover]

            # Find cameras on this rover that support the mode
            suitable_cameras = []
            for cam, r in self.camera_on_rover.items():
                if r == rover and m in self.camera_modes.get(cam, set()):
                    suitable_cameras.append(cam)

            if not suitable_cameras: continue # Rover cannot take this image

            for camera in suitable_cameras:
                cost_to_get_data = 0
                # Check if this rover already has the data
                if (o, m) not in have_image.get(rover, set()):
                    # Need to take image

                    # Cost to navigate to an image waypoint
                    min_nav_to_img_cost = float('inf')
                    best_img_wp = None
                    for img_wp in img_wps:
                        nav_cost = self.get_shortest_path_cost(rover, current_wp, img_wp)
                        if nav_cost < min_nav_to_img_cost:
                            min_nav_to_img_cost = nav_cost
                            best_img_wp = img_wp

                    if min_nav_to_img_cost == float('inf'): continue # Cannot reach any image point

                    cost_to_get_data += min_nav_to_img_cost # navigate

                    # Cost to calibrate if needed
                    if not camera_calibrated.get(camera, False):
                        cal_target = self.camera_cal_target.get(camera)
                        if not cal_target: continue # Cannot calibrate this camera

                        cal_wps = self.caltarget_visible_wps.get(cal_target, set())
                        if not cal_wps: continue # Cannot calibrate target

                        # Cost to navigate to a calibration waypoint (from current_wp)
                        min_nav_to_cal_cost = float('inf')
                        best_cal_wp = None
                        for cal_wp in cal_wps:
                             nav_cost = self.get_shortest_path_cost(rover, current_wp, cal_wp)
                             if nav_cost < min_nav_to_cal_cost:
                                 min_nav_to_cal_cost = nav_cost
                                 best_cal_wp = cal_wp

                        if min_nav_to_cal_cost == float('inf'): continue # Cannot reach any calibration point

                        cost_to_get_data += min_nav_to_cal_cost + 1 # navigate + calibrate

                    cost_to_get_data += 1 # take_image action

                    # After taking image, rover is at best_img_wp
                    location_after_data = best_img_wp
                else:
                    # Rover already has data
                    location_after_data = current_wp # Rover is still at its current location

                # Cost to communicate
                min_nav_to_comm_cost = float('inf')
                for comm_wp in comm_wps:
                    nav_cost = self.get_shortest_path_cost(rover, location_after_data, comm_wp)
                    min_nav_to_comm_cost = min(min_nav_to_comm_cost, nav_cost)

                if min_nav_to_comm_cost == float('inf'): continue # Cannot reach communication point

                comm_cost = min_nav_to_comm_cost + 1 # navigate + communicate

                total_cost_for_this_rover_camera = cost_to_get_data + comm_cost
                min_total_cost = min(min_total_cost, total_cost_for_this_rover_camera)

        return min_total_cost if min_total_cost != float('inf') else 1000 # Use a large finite value


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state

        # Goal check
        if self.goals <= state:
            return 0

        # --- Parse Dynamic State Information ---
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        have_soil = {} # rover -> set of waypoints
        have_rock = {} # rover -> set of waypoints
        have_image = {} # rover -> set of (objective, mode) tuples
        camera_calibrated = {} # camera -> True/False

        for fact_str in state:
            parts = get_parts(fact_str)
            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3 and parts[1].startswith('rover'):
                rover, wp = parts[1], parts[2]
                rover_locations[rover] = wp
            elif predicate == 'empty' and len(parts) == 2 and parts[1].startswith('rover') and parts[1].endswith('store'):
                store = parts[1]
                store_status[store] = 'empty'
            elif predicate == 'full' and len(parts) == 2 and parts[1].startswith('rover') and parts[1].endswith('store'):
                store = parts[1]
                store_status[store] = 'full'
            elif predicate == 'have_soil_analysis' and len(parts) == 3 and parts[1].startswith('rover'):
                rover, wp = parts[1], parts[2]
                have_soil.setdefault(rover, set()).add(wp)
            elif predicate == 'have_rock_analysis' and len(parts) == 3 and parts[1].startswith('rover'):
                rover, wp = parts[1], parts[2]
                have_rock.setdefault(rover, set()).add(wp)
            elif predicate == 'have_image' and len(parts) == 4 and parts[1].startswith('rover'):
                rover, obj, mode = parts[1], parts[2], parts[3]
                have_image.setdefault(rover, set()).add((obj, mode))
            elif predicate == 'calibrated' and len(parts) == 3 and parts[1].startswith('camera'):
                camera, rover = parts[1], parts[2]
                camera_calibrated[camera] = True

        # For rovers not found in state 'at' facts, assume they are at their initial location
        for rover, initial_wp in self.initial_rover_locations.items():
            if rover not in rover_locations:
                rover_locations[rover] = initial_wp

        # Add uncalibrated cameras (all cameras not in camera_calibrated)
        all_cameras = set(self.camera_on_rover.keys())
        for cam in all_cameras:
            if cam not in camera_calibrated:
                camera_calibrated[cam] = False


        state_info = {
            'rover_locations': rover_locations,
            'store_status': store_status,
            'have_soil': have_soil,
            'have_rock': have_rock,
            'have_image': have_image,
            'camera_calibrated': camera_calibrated,
        }

        h = 0
        # Iterate through unsatisfied goals
        for goal in self.goals:
            if goal not in state:
                parts = get_parts(goal)
                predicate = parts[0]

                cost_for_this_goal = 0
                if predicate == 'communicated_soil_data' and len(parts) == 2:
                    w = parts[1]
                    cost_for_this_goal = self.cost_soil_goal(state_info, w)
                elif predicate == 'communicated_rock_data' and len(parts) == 2:
                    w = parts[1]
                    cost_for_this_goal = self.cost_rock_goal(state_info, w)
                elif predicate == 'communicated_image_data' and len(parts) == 3: # (communicated_image_data obj mode)
                    o, m = parts[1], parts[2]
                    cost_for_this_goal = self.cost_image_goal(state_info, o, m)
                else:
                    # Unknown goal type, or malformed goal
                    # This shouldn't happen in valid PDDL, but as a fallback
                    # treat as impossible or add a high cost.
                    cost_for_this_goal = 1000 # Treat as impossible

                h += cost_for_this_goal

                # If any goal is impossible, the total heuristic should reflect that
                if h >= 1000:
                     return 1000

        return h
