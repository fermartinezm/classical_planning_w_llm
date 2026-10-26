from heuristics.heuristic_base import Heuristic
from task import Operator, Task
import collections

# Helper function to parse PDDL fact strings
def parse_fact(fact_str):
    """Parses a PDDL fact string into a tuple."""
    # Removes parentheses and splits by space
    parts = fact_str[1:-1].split()
    return tuple(parts)

# Helper function for BFS
def bfs(graph, start_node):
    """Computes shortest distances from start_node in a graph."""
    distances = {node: float('inf') for node in graph}
    # Ensure start_node is in the graph keys, even if it has no outgoing edges
    if start_node not in graph:
         graph[start_node] = set()

    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        if current_node in graph: # Should be true
            for neighbor in graph.get(current_node, set()): # Use .get for safety
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest distances between all pairs of nodes in a graph."""
    all_paths = {}
    # Collect all nodes (waypoints) from graph keys and values
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    # Ensure all nodes are in the graph keys for BFS start
    for node in all_nodes:
         graph.setdefault(node, set())

    for start_node in all_nodes:
         all_paths[start_node] = bfs(graph, start_node)
    return all_paths


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up the
    estimated costs for each unachieved goal fact. For each unachieved goal,
    it finds the minimum cost required to satisfy it, considering available
    rovers, necessary locations (sample sites, image waypoints, calibration
    waypoints, communication waypoints), and required actions (navigate,
    sample, drop, calibrate, take_image, communicate). Navigation costs are
    precomputed using BFS. The heuristic is non-admissible and aims to guide
    a greedy best-first search by prioritizing states closer to satisfying
    multiple goals or critical subgoals.

    Assumptions:
    - The PDDL domain and problem files are well-formed according to the
      provided structure.
    - All objects referenced in facts exist.
    - Navigation is possible between waypoints if 'can_traverse' is true for
      the specific rover and direction.
    - The heuristic assumes that if a sample goal is not met and the sample
      is no longer at its initial location, it must have been collected by
      a rover (and checks if any rover 'have' it). If the sample is gone
      and no rover 'have' it, the goal is considered impossible.
    - Calibration cost for an image goal is estimated based on navigating from
      the image waypoint to the nearest calibration waypoint for the camera,
      plus the calibrate action. This is a simplification; in reality, the
      rover might need to navigate elsewhere to calibrate before returning
      to the image waypoint.

    Heuristic Initialization:
    The constructor pre-processes the static information from the task:
    - Parses all static facts to identify rover capabilities, camera properties,
      waypoint relationships (visibility for communication, imaging, calibration),
      and lander location.
    - Builds navigation graphs for each rover based on 'can_traverse' facts.
    - Computes all-pairs shortest paths for each rover's navigation graph using BFS.
    - Stores initial locations of soil and rock samples.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state to extract dynamic information: rover locations,
        locations of remaining samples, which rovers have collected which data,
        which cameras are calibrated, and which stores are full.
    2.  Initialize the total heuristic value `h` to 0.
    3.  Iterate through each goal fact specified in the task.
    4.  For an unachieved goal fact:
        a.  Determine the type of goal (soil, rock, or image communication).
        b.  Estimate the minimum cost to achieve this specific goal fact from the
            current state. This involves considering:
            i.  Whether the required data (soil sample, rock sample, image) is
                already collected by any rover.
            ii. If data is already collected: Estimate the minimum cost for a
                rover holding the data to navigate from its current location to a
                communication waypoint and perform the communication action
                (navigation cost + 1). Minimize this over suitable rovers and
                communication waypoints.
            iii. If data is not collected: Estimate the minimum cost to first
                collect the data and then communicate it. This involves finding
                a suitable rover and location:
                -   **For soil/rock:** Check if the sample is still at its original
                    waypoint. If yes, find an equipped rover, estimate cost to
                    navigate it to the sample location, perform sample action
                    (consider drop action if store is full), navigate from sample
                    location to a communication waypoint, and communicate.
                    Estimate cost = Nav(current to sample) + (1 if drop) + 1 (sample)
                                  + Nav(sample to comm_wp) + 1 (communicate).
                    Minimize this over suitable rovers and communication waypoints.
                    If the sample is gone from its waypoint AND no rover has the
                    data, the goal is considered impossible (cost infinity).
                -   **For image:** Find an equipped rover with a suitable camera,
                    find a waypoint visible from the objective, estimate cost to
                    navigate the rover to that waypoint, estimate cost to perform
                    calibration (if needed), take the image, navigate from the
                    image waypoint to a communication waypoint, and communicate.
                    Estimate cost = Nav(current to image_wp)
                                  + Cost(calibration) + 1 (take_image)
                                  + Nav(image_wp to comm_wp) + 1 (communicate).
                    Cost(calibration) = 0 if camera is calibrated. If not, it's
                    estimated as Nav(image_wp to nearest cal_wp) + 1 (calibrate).
                    Minimize this over suitable rovers, cameras, image waypoints,
                    and communication waypoints. If no suitable path or resource
                    exists, the cost is infinity.
        c.  If the minimum estimated cost for the goal fact is infinity, the entire
            problem is likely unsolvable from this state, so return infinity immediately.
        d.  Add the minimum estimated cost for the goal fact to the total heuristic `h`.
    5.  Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goals = task.goals

        # --- Preprocess Static Information ---
        self.rover_types = {}  # rover -> set(type_str) e.g., {'rover1': {'soil', 'rock', 'imaging'}}
        self.rover_stores = {} # rover -> store
        self.rover_cameras = {} # rover -> set(camera)
        self.camera_modes = {} # camera -> set(mode)
        self.camera_cal_target = {} # camera -> objective
        self.obj_image_wps = {} # objective -> set(waypoint)
        self.caltarget_cal_wps = {} # objective -> set(waypoint)
        self.lander_wp = None
        self.comm_wps = set() # waypoints visible from lander_wp
        self.rover_nav_graphs = {} # rover -> {wp: set(neighbor_wp)}
        self.waypoints = set() # Collect all waypoints

        # Initial sample locations (static)
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()

        # Collect all waypoints mentioned in static facts first
        for fact_str in task.static:
             fact = parse_fact(fact_str)
             for item in fact[1:]:
                 if isinstance(item, str) and item.startswith('waypoint'): # Simple check
                     self.waypoints.add(item)

        # Also collect waypoints from initial state
        for fact_str in task.initial_state:
             fact = parse_fact(fact_str)
             for item in fact[1:]:
                 if isinstance(item, str) and item.startswith('waypoint'): # Simple check
                     self.waypoints.add(item)

        # Parse static facts and build structures
        for fact_str in task.static:
            fact = parse_fact(fact_str)
            if fact[0] == 'equipped_for_soil_analysis':
                rover = fact[1]
                self.rover_types.setdefault(rover, set()).add('soil')
            elif fact[0] == 'equipped_for_rock_analysis':
                rover = fact[1]
                self.rover_types.setdefault(rover, set()).add('rock')
            elif fact[0] == 'equipped_for_imaging':
                rover = fact[1]
                self.rover_types.setdefault(rover, set()).add('imaging')
            elif fact[0] == 'store_of':
                store, rover = fact[1], fact[2]
                self.rover_stores[rover] = store
            elif fact[0] == 'on_board':
                camera, rover = fact[1], fact[2]
                self.rover_cameras.setdefault(rover, set()).add(camera)
            elif fact[0] == 'supports':
                camera, mode = fact[1], fact[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif fact[0] == 'calibration_target':
                camera, objective = fact[1], fact[2]
                self.camera_cal_target[camera] = objective
            elif fact[0] == 'at_lander':
                lander, wp = fact[1], fact[2]
                self.lander_wp = wp
            elif fact[0] == 'visible':
                 wp1, wp2 = fact[1], fact[2]
                 # Comm waypoints are visible from lander_wp
                 if self.lander_wp and wp2 == self.lander_wp:
                     self.comm_wps.add(wp1)
            elif fact[0] == 'can_traverse':
                rover, wp1, wp2 = fact[1], fact[2], fact[3]
                self.rover_nav_graphs.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)
            elif fact[0] == 'visible_from':
                 obj, wp = fact[1], fact[2]
                 self.obj_image_wps.setdefault(obj, set()).add(wp)
                 # Check if this objective is a calibration target for any camera
                 is_cal_target = False
                 for cam, target_obj in self.camera_cal_target.items():
                     if target_obj == obj:
                         self.caltarget_cal_wps.setdefault(obj, set()).add(wp)
                         is_cal_target = True
                         break # Assume an objective is only a cal target for one camera?

        # Ensure all waypoints are in the graph keys for BFS start for each rover
        for rover in self.rover_nav_graphs:
             for wp in self.waypoints:
                 self.rover_nav_graphs[rover].setdefault(wp, set())

        # Compute all-pairs shortest paths for each rover
        self.rover_shortest_paths = {}
        for rover, graph in self.rover_nav_graphs.items():
            self.rover_shortest_paths[rover] = compute_all_pairs_shortest_paths(graph)

        # Parse initial state for initial sample locations
        for fact_str in task.initial_state:
            fact = parse_fact(fact_str)
            if fact[0] == 'at_soil_sample':
                self.initial_soil_samples.add(fact[1])
            elif fact[0] == 'at_rock_sample':
                self.initial_rock_samples.add(fact[1])


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # --- Parse Dynamic Information from State ---
        rover_locations = {} # rover -> waypoint
        soil_samples_at = set() # waypoints with soil samples
        rock_samples_at = set() # waypoints with rock samples
        rover_soil_data = {} # rover -> set(waypoint)
        rover_rock_data = {} # rover -> set(waypoint)
        rover_image_data = {} # rover -> set((objective, mode))
        rover_calibrated_cameras = {} # rover -> set(camera)
        rover_full_stores = set() # stores that are full

        for fact_str in state:
            fact = parse_fact(fact_str)
            if fact[0] == 'at':
                rover, wp = fact[1], fact[2]
                rover_locations[rover] = wp
            elif fact[0] == 'at_soil_sample':
                soil_samples_at.add(fact[1])
            elif fact[0] == 'at_rock_sample':
                rock_samples_at.add(fact[1])
            elif fact[0] == 'have_soil_analysis':
                rover, wp = fact[1], fact[2]
                rover_soil_data.setdefault(rover, set()).add(wp)
            elif fact[0] == 'have_rock_analysis':
                rover, wp = fact[1], fact[2]
                rover_rock_data.setdefault(rover, set()).add(wp)
            elif fact[0] == 'have_image':
                rover, obj, mode = fact[1], fact[2], fact[3]
                rover_image_data.setdefault(rover, set()).add((obj, mode))
            elif fact[0] == 'calibrated':
                camera, rover = fact[1], fact[2]
                rover_calibrated_cameras.setdefault(rover, set()).add(camera)
            elif fact[0] == 'full':
                store = fact[1]
                rover_full_stores.add(store)

        # --- Compute Heuristic ---
        h = 0

        for goal_fact_str in self.goals:
            if goal_fact_str in state:
                continue # Goal already achieved

            goal_fact = parse_fact(goal_fact_str)
            goal_type = goal_fact[0]

            goal_cost = float('inf')

            if goal_type == 'communicated_soil_data':
                w = goal_fact[1]
                # Case 1: Data is already collected by a rover
                min_cost_if_have = float('inf')
                found_rover_with_data = False
                for r in rover_soil_data:
                    if w in rover_soil_data.get(r, set()):
                        found_rover_with_data = True
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue # Rover location unknown? Should not happen in valid state.

                        min_comm_nav = float('inf')
                        for comm_wp in self.comm_wps:
                            if current_wp in self.rover_shortest_paths.get(r, {}) and comm_wp in self.rover_shortest_paths[r].get(current_wp, {}):
                                min_comm_nav = min(min_comm_nav, self.rover_shortest_paths[r][current_wp][comm_wp])

                        if min_comm_nav != float('inf'):
                            min_cost_if_have = min(min_cost_if_have, min_comm_nav + 1) # +1 for communicate action

                # Case 2: Data needs to be sampled and then communicated
                min_cost_if_sample = float('inf')
                # Check if sample is available to be sampled (i.e., still at the waypoint)
                sample_still_at_wp = w in soil_samples_at

                if sample_still_at_wp:
                    for r in self.rover_types:
                        if 'soil' in self.rover_types.get(r, set()):
                            current_wp = rover_locations.get(r)
                            if current_wp is None: continue

                            if current_wp in self.rover_shortest_paths.get(r, {}) and w in self.rover_shortest_paths[r].get(current_wp, {}):
                                nav_cost_to_sample = self.rover_shortest_paths[r][current_wp][w]
                                store = self.rover_stores.get(r)
                                drop_cost = 1 if store in rover_full_stores else 0
                                sample_action_cost = 1
                                cost_get_sample = nav_cost_to_sample + drop_cost + sample_action_cost

                                min_comm_nav_from_sample_wp = float('inf')
                                for comm_wp in self.comm_wps:
                                    if w in self.rover_shortest_paths.get(r, {}) and comm_wp in self.rover_shortest_paths[r].get(w, {}):
                                         min_comm_nav_from_sample_wp = min(min_comm_nav_from_sample_wp, self.rover_shortest_paths[r][w][comm_wp])

                                if min_comm_nav_from_sample_wp != float('inf'):
                                    min_cost_if_sample = min(min_cost_if_sample, cost_get_sample + min_comm_nav_from_sample_wp + 1) # +1 for communicate action

                # The cost for this goal is the minimum of having it or sampling it
                goal_cost = min(min_cost_if_have, min_cost_if_sample)

                # If sample is gone AND no one has it, goal is impossible
                if not sample_still_at_wp and not found_rover_with_data:
                     goal_cost = float('inf') # Impossible

            elif goal_type == 'communicated_rock_data':
                w = goal_fact[1]
                # Case 1: Data is already collected by a rover
                min_cost_if_have = float('inf')
                found_rover_with_data = False
                for r in rover_rock_data:
                    if w in rover_rock_data.get(r, set()):
                        found_rover_with_data = True
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue

                        min_comm_nav = float('inf')
                        for comm_wp in self.comm_wps:
                            if current_wp in self.rover_shortest_paths.get(r, {}) and comm_wp in self.rover_shortest_paths[r].get(current_wp, {}):
                                min_comm_nav = min(min_comm_nav, self.rover_shortest_paths[r][current_wp][comm_wp])

                        if min_comm_nav != float('inf'):
                            min_cost_if_have = min(min_cost_if_have, min_comm_nav + 1) # +1 for communicate action

                # Case 2: Data needs to be sampled and then communicated
                min_cost_if_sample = float('inf')
                sample_still_at_wp = w in rock_samples_at

                if sample_still_at_wp:
                    for r in self.rover_types:
                        if 'rock' in self.rover_types.get(r, set()):
                            current_wp = rover_locations.get(r)
                            if current_wp is None: continue

                            if current_wp in self.rover_shortest_paths.get(r, {}) and w in self.rover_shortest_paths[r].get(current_wp, {}):
                                nav_cost_to_sample = self.rover_shortest_paths[r][current_wp][w]
                                store = self.rover_stores.get(r)
                                drop_cost = 1 if store in rover_full_stores else 0
                                sample_action_cost = 1
                                cost_get_sample = nav_cost_to_sample + drop_cost + sample_action_cost

                                min_comm_nav_from_sample_wp = float('inf')
                                for comm_wp in self.comm_wps:
                                    if w in self.rover_shortest_paths.get(r, {}) and comm_wp in self.rover_shortest_paths[r].get(w, {}):
                                         min_comm_nav_from_sample_wp = min(min_comm_nav_from_sample_wp, self.rover_shortest_paths[r][w][comm_wp])

                                if min_comm_nav_from_sample_wp != float('inf'):
                                    min_cost_if_sample = min(min_cost_if_sample, cost_get_sample + min_comm_nav_from_sample_wp + 1) # +1 for communicate action

                # The cost for this goal is the minimum of having it or sampling it
                goal_cost = min(min_cost_if_have, min_cost_if_sample)

                # If sample is gone AND no one has it, goal is impossible
                if not sample_still_at_wp and not found_rover_with_data:
                     goal_cost = float('inf') # Impossible


            elif goal_type == 'communicated_image_data':
                o, m = goal_fact[1], goal_fact[2]
                # Case 1: Data is already collected by a rover
                min_cost_if_have = float('inf')
                found_rover_with_data = False
                for r in rover_image_data:
                    if (o, m) in rover_image_data.get(r, set()):
                        found_rover_with_data = True
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue

                        min_comm_nav = float('inf')
                        for comm_wp in self.comm_wps:
                            if current_wp in self.rover_shortest_paths.get(r, {}) and comm_wp in self.rover_shortest_paths[r].get(current_wp, {}):
                                min_comm_nav = min(min_comm_nav, self.rover_shortest_paths[r][current_wp][comm_wp])

                        if min_comm_nav != float('inf'):
                            min_cost_if_have = min(min_cost_if_have, min_comm_nav + 1) # +1 for communicate action

                # Case 2: Data needs to be imaged and then communicated
                min_cost_if_image = float('inf')
                image_wps = self.obj_image_wps.get(o, set())

                if not image_wps: # No waypoint to view objective from
                     # min_cost_if_image remains infinity
                     pass
                else:
                    for r in self.rover_types:
                        if 'imaging' in self.rover_types.get(r, set()):
                            current_wp = rover_locations.get(r)
                            if current_wp is None: continue

                            for camera in self.rover_cameras.get(r, set()):
                                if m in self.camera_modes.get(camera, set()):
                                    # Found a suitable rover/camera/mode combo
                                    min_cost_for_combo = float('inf')
                                    for image_wp in image_wps:
                                        if current_wp in self.rover_shortest_paths.get(r, {}) and image_wp in self.rover_shortest_paths[r].get(current_wp, {}):
                                            nav_cost_to_image_wp = self.rover_shortest_paths[r][current_wp][image_wp]
                                            take_image_action_cost = 1

                                            cal_cost = 0
                                            if camera not in rover_calibrated_cameras.get(r, set()):
                                                # Need to calibrate
                                                cal_target = self.camera_cal_target.get(camera)
                                                if cal_target:
                                                    cal_wps = self.caltarget_cal_wps.get(cal_target, set())
                                                    if not cal_wps: # No waypoint to view calibration target from
                                                        cal_cost = float('inf') # Impossible for this camera/target
                                                    else:
                                                        min_cal_nav_from_image_wp = float('inf')
                                                        # Calibration happens from the image_wp or nearest cal_wp from image_wp
                                                        if image_wp in self.rover_shortest_paths.get(r, {}):
                                                            for cal_wp in cal_wps:
                                                                if cal_wp in self.rover_shortest_paths[r].get(image_wp, {}):
                                                                    min_cal_nav_from_image_wp = min(min_cal_nav_from_image_wp, self.rover_shortest_paths[r][image_wp][cal_wp])

                                                        if min_cal_nav_from_image_wp != float('inf'):
                                                            cal_cost = min_cal_nav_from_image_wp + 1 # +1 for calibrate action
                                                        else:
                                                            cal_cost = float('inf') # Cannot reach any cal wp from image wp
                                                else: # Camera has no calibration target defined
                                                     # PDDL requires calibration_target for calibrate action.
                                                     # If no target, calibrate is impossible.
                                                     cal_cost = float('inf') # Cannot calibrate

                                            if cal_cost != float('inf'):
                                                min_comm_nav_from_image_wp = float('inf')
                                                if image_wp in self.rover_shortest_paths.get(r, {}):
                                                    for comm_wp in self.comm_wps:
                                                        if comm_wp in self.rover_shortest_paths[r].get(image_wp, {}):
                                                            min_comm_nav_from_image_wp = min(min_comm_nav_from_image_wp, self.rover_shortest_paths[r][image_wp][comm_wp])

                                                if min_comm_nav_from_image_wp != float('inf'):
                                                    cost_get_image = nav_cost_to_image_wp + cal_cost + take_image_action_cost
                                                    min_cost_for_combo = min(min_cost_for_combo, cost_get_image + min_comm_nav_from_image_wp + 1) # +1 for communicate action

                                    if min_cost_for_combo != float('inf'):
                                         min_cost_if_image = min(min_cost_if_image, min_cost_for_combo)

                # The cost for this goal is the minimum of having it or imaging it
                goal_cost = min(min_cost_if_have, min_cost_if_image)


            # If after checking all options, the goal is still impossible, return infinity
            if goal_cost == float('inf'):
                return float('inf')

            h += goal_cost

        return h
