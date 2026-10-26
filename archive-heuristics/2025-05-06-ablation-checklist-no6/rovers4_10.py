import collections
import fnmatch
import math # Import math for infinity

# Assuming Heuristic base class is available at this path
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle facts like "(at rover1 waypoint1)" or "(communicated_soil_data waypoint4)"
    # Just remove outer parentheses and split by space.
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards `*` allowed.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function for Breadth-First Search to find shortest paths
def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {start_node: 0}
    queue = collections.deque([start_node])
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        # Check if current_node has outgoing edges in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances

# Define the domain-dependent heuristic class
# class roversHeuristic(Heuristic): # Uncomment and inherit if the base class is available
class roversHeuristic: # Use this if the base class is not provided directly
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It decomposes the problem into individual goals
    (communicating soil, rock, or image data) and estimates the minimum cost
    to achieve each one independently, summing these minimum costs.

    # Heuristic Components
    For each unachieved communication goal (soil, rock, or image):
    1.  **Check if data is already held:** If a rover currently possesses the required data,
        estimate the cost to navigate to a communication point and communicate.
    2.  **Estimate cost to acquire data and communicate:** If no rover has the data,
        estimate the cost for a capable rover to:
        a.  Navigate to the location where data can be acquired (sample site, image waypoint).
        b.  Perform the acquisition action (sample, take_image). This may involve
            prerequisites like having an empty store (for samples) or calibrating
            a camera (for images). Simplified costs are added for these prerequisites.
        c.  Navigate from the acquisition location to a communication point.
        d.  Perform the communication action.
    The heuristic takes the minimum cost between option 1 and option 2 (if applicable)
    for each goal and sums these minimum costs.

    # Navigation Costs
    Shortest path distances between waypoints for each rover are precomputed
    using BFS on the `can_traverse` graph.

    # Simplifications (Non-Admissible)
    - Ignores negative interactions between goals (e.g., using a store for one sample
      prevents using it for another until dropped).
    - Simplified cost for dropping a sample (adds 1 action if a store is full and sampling is needed).
    - Simplified cost for camera calibration (adds navigation to calibration point + 1 action,
      ignores navigation back to image point).
    - Assumes communication can happen from any waypoint visible from the lander.
    - Sums costs of individual goals, ignoring potential synergies (e.g., navigating to a
      waypoint that serves multiple purposes).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts,
        and precomputing navigation distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # Data structures to store static information
        self.lander_location = None
        self.rover_equipment = collections.defaultdict(dict) # {rover: {equipment_type: True}}
        self.store_of = {} # {store: rover}
        self.initial_soil_samples = set() # {waypoint}
        self.initial_rock_samples = set() # {waypoint}
        self.on_board = {} # {camera: rover}
        self.supports = collections.defaultdict(set) # {camera: {mode}}
        self.calibration_target = {} # {camera: objective}
        self.visible_from = collections.defaultdict(set) # {objective/target: {waypoint}}
        self.rover_graphs = collections.defaultdict(lambda: collections.defaultdict(list)) # {rover: {wp: [neighbor_wp]}}
        self.all_waypoints = set() # Collect all waypoints

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            pred = parts[0]
            if pred == 'at_lander':
                self.lander_location = parts[2]
            elif pred.startswith('equipped_for_'):
                rover = parts[1]
                equipment_type = pred[len('equipped_for_'):] # 'soil_analysis', 'rock_analysis', 'imaging'
                self.rover_equipment[rover][equipment_type] = True
            elif pred == 'store_of':
                store, rover = parts[1], parts[2]
                self.store_of[store] = rover
            elif pred == 'at_soil_sample':
                self.initial_soil_samples.add(parts[1])
            elif pred == 'at_rock_sample':
                self.initial_rock_samples.add(parts[1])
            elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                self.on_board[camera] = rover
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                self.supports[camera].add(mode)
            elif pred == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.calibration_target[camera] = objective
            elif pred == 'visible_from':
                objective_target, waypoint = parts[1], parts[2]
                self.visible_from[objective_target].add(waypoint)
            elif pred == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                self.rover_graphs[rover][wp1].append(wp2)
                self.all_waypoints.add(wp1)
                self.all_waypoints.add(wp2)
            # Note: 'visible' predicate is used for communication and imaging/calibration,
            # but not directly for rover navigation graph according to action preconditions.
            # We parse 'visible_from' and use 'visible' with lander location below.
            # Other predicates like 'at', 'empty', 'full', etc., are state-dependent.

        # Ensure all waypoints mentioned in visible_from are included even if not in can_traverse
        for wps in self.visible_from.values():
             self.all_waypoints.update(wps)
        # Ensure lander location is included
        if self.lander_location:
             self.all_waypoints.add(self.lander_location)

        # Precompute shortest paths for each rover
        self.rover_distances = {}
        for rover, graph in self.rover_graphs.items():
            self.rover_distances[rover] = {}
            # Run BFS from every waypoint that is part of this rover's graph
            reachable_waypoints = set(graph.keys()).union(set(wp for neighbors in graph.values() for wp in neighbors))
            for start_wp in reachable_waypoints:
                 self.rover_distances[rover][start_wp] = bfs(graph, start_wp)

        # Identify communication waypoints (visible from lander location)
        self.comm_waypoints = set()
        if self.lander_location:
             # The communicate action requires (visible ?x ?y) where rover is at ?x and lander at ?y.
             # We need waypoints ?x such that (visible ?x lander_location) is true.
             for fact in static_facts:
                 if match(fact, 'visible', '*', self.lander_location):
                     self.comm_waypoints.add(get_parts(fact)[1])

        # Map camera to calibration waypoints
        self.cal_waypoints = {} # {camera: {waypoint}}
        for camera, target in self.calibration_target.items():
             self.cal_waypoints[camera] = self.visible_from.get(target, set())

        # Map objective to image waypoints
        self.image_waypoints = {} # {objective: {waypoint}}
        # Filter visible_from for objectives mentioned in goals
        goal_objectives = set()
        for goal in self.goals:
             if match(goal, 'communicated_image_data', '*', '*'):
                 goal_objectives.add(get_parts(goal)[1])

        for obj in goal_objectives:
             if obj in self.visible_from:
                 # An image goal requires a specific mode, but the visible_from is per objective.
                 # Any waypoint visible from the objective works for any mode supported by the camera.
                 # So, image waypoints are per objective.
                 self.image_waypoints[obj] = self.visible_from[obj]


    def get_distance(self, rover, start_wp, end_wp):
        """Get shortest path distance for a rover between two waypoints."""
        # Return infinity if rover or waypoints are not in the precomputed distances
        if rover not in self.rover_distances or start_wp not in self.rover_distances[rover] or end_wp not in self.rover_distances[rover][start_wp]:
            return math.inf
        return self.rover_distances[rover][start_wp][end_wp]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Parse current state to get dynamic facts
        current_rover_locations = {}
        rover_have_soil = set() # {(rover, waypoint)}
        rover_have_rock = set() # {(rover, waypoint)}
        rover_have_image = set() # {(rover, objective, mode)}
        communicated_soil = set() # {waypoint}
        communicated_rock = set() # {waypoint}
        communicated_image = set() # {(objective, mode)}
        rover_stores_full = collections.defaultdict(bool) # {rover: True/False}
        camera_calibrated = set() # {(camera, rover)}
        current_soil_samples = set() # {waypoint}
        current_rock_samples = set() # {waypoint}


        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at':
                obj, loc = parts[1], parts[2]
                # Assuming only rovers and landers have 'at' predicate relevant here
                if obj.startswith('rover'): # Simple check for rover type
                     current_rover_locations[obj] = loc
            elif pred == 'have_soil_analysis':
                rover, wp = parts[1], parts[2]
                rover_have_soil.add((rover, wp))
            elif pred == 'have_rock_analysis':
                rover, wp = parts[1], parts[2]
                rover_have_rock.add((rover, wp))
            elif pred == 'have_image':
                rover, obj, mode = parts[1], parts[2], parts[3]
                rover_have_image.add((rover, obj, mode))
            elif pred == 'communicated_soil_data':
                communicated_soil.add(parts[1])
            elif pred == 'communicated_rock_data':
                communicated_rock.add(parts[1])
            elif pred == 'communicated_image_data':
                communicated_image.add((parts[1], parts[2]))
            elif pred == 'full':
                 store = parts[1]
                 if store in self.store_of:
                     rover = self.store_of[store]
                     rover_stores_full[rover] = True # Mark rover as having at least one full store
            elif pred == 'calibrated':
                 camera, rover = parts[1], parts[2]
                 camera_calibrated.add((camera, rover))
            elif pred == 'at_soil_sample':
                current_soil_samples.add(parts[1])
            elif pred == 'at_rock_sample':
                current_rock_samples.add(parts[1])


        total_cost = 0

        # Iterate through goals and sum up costs for unachieved ones
        for goal in self.goals:
            # Check if the goal fact is already true in the current state
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            pred = parts[0]

            min_goal_cost = math.inf # Minimum cost to achieve this specific goal

            if pred == 'communicated_soil_data':
                wp = parts[1]

                # Option B: Communicate data already held by a rover
                min_cost_comm_only = math.inf
                for rover, data_wp in rover_have_soil:
                    if data_wp == wp:
                        # This rover has the required soil data
                        cost = 0
                        # Cost to communicate from rover's current location
                        min_comm_nav_cost = math.inf
                        if self.comm_waypoints: # Check if there are any communication waypoints
                            min_comm_nav_cost = min(self.get_distance(rover, current_rover_locations[rover], comm_wp) for comm_wp in self.comm_waypoints)

                        if min_comm_nav_cost == math.inf:
                             cost = math.inf # Cannot reach any communication point
                        else:
                             cost += min_comm_nav_cost
                             cost += 1 # communicate action
                        min_cost_comm_only = min(min_cost_comm_only, cost) # Min over rovers holding the data

                min_goal_cost = min(min_goal_cost, min_cost_comm_only)


                # Option A: Get data (sample) then communicate
                # Can only sample if the sample is still at the waypoint
                if wp in current_soil_samples:
                    # Find capable rovers
                    capable_rovers = [r for r in current_rover_locations if self.rover_equipment.get(r, {}).get('soil_analysis', False)]
                    if capable_rovers:
                        for rover in capable_rovers:
                            cost = 0
                            # Cost to get sample at wp
                            cost_get_sample = self.get_distance(rover, current_rover_locations[rover], wp) # Nav to sample point
                            cost_get_sample += 1 # sample_soil action
                            # Simplified store cost: add 1 if rover has any full store
                            rover_stores = [s for s, rv in self.store_of.items() if rv == rover]
                            if any(f'(full {s})' in state for s in rover_stores):
                                cost_get_sample += 1 # drop action

                            # Cost to communicate from wp (after sampling)
                            min_comm_nav = math.inf
                            if self.comm_waypoints: # Check if there are any communication waypoints
                                min_comm_nav = min(self.get_distance(rover, wp, comm_wp) for comm_wp in self.comm_waypoints)

                            if min_comm_nav == math.inf:
                                cost = math.inf # Cannot reach any communication point from sample location
                            else:
                                cost = cost_get_sample + min_comm_nav + 1 # Nav to comm + Communicate action

                            min_goal_cost = min(min_goal_cost, cost) # Min over capable rovers

            elif pred == 'communicated_rock_data':
                wp = parts[1]

                # Option B: Communicate data already held by a rover
                min_cost_comm_only = math.inf
                for rover, data_wp in rover_have_rock:
                    if data_wp == wp:
                        # This rover has the required rock data
                        cost = 0
                        # Cost to communicate from rover's current location
                        min_comm_nav_cost = math.inf
                        if self.comm_waypoints: # Check if there are any communication waypoints
                            min_comm_nav_cost = min(self.get_distance(rover, current_rover_locations[rover], comm_wp) for comm_wp in self.comm_waypoints)

                        if min_comm_nav_cost == math.inf:
                             cost = math.inf # Cannot reach any communication point
                        else:
                             cost += min_comm_nav_cost
                             cost += 1 # communicate action
                        min_cost_comm_only = min(min_cost_comm_only, cost) # Min over rovers holding the data

                min_goal_cost = min(min_goal_cost, min_cost_comm_only)

                # Option A: Get data (sample) then communicate
                # Can only sample if the sample is still at the waypoint
                if wp in current_rock_samples:
                    # Find capable rovers
                    capable_rovers = [r for r in current_rover_locations if self.rover_equipment.get(r, {}).get('rock_analysis', False)]
                    if capable_rovers:
                        for rover in capable_rovers:
                            cost = 0
                            # Cost to get sample at wp
                            cost_get_sample = self.get_distance(rover, current_rover_locations[rover], wp) # Nav to sample point
                            cost_get_sample += 1 # sample_rock action
                            # Simplified store cost: add 1 if rover has any full store
                            rover_stores = [s for s, rv in self.store_of.items() if rv == rover]
                            if any(f'(full {s})' in state for s in rover_stores):
                                cost_get_sample += 1 # drop action

                            # Cost to communicate from wp (after sampling)
                            min_comm_nav = math.inf
                            if self.comm_waypoints: # Check if there are any communication waypoints
                                min_comm_nav = min(self.get_distance(rover, wp, comm_wp) for comm_wp in self.comm_waypoints)

                            if min_comm_nav == math.inf:
                                cost = math.inf # Cannot reach any communication point from sample location
                            else:
                                cost = cost_get_sample + min_comm_nav + 1 # Nav to comm + Communicate action

                            min_goal_cost = min(min_goal_cost, cost) # Min over capable rovers


            elif pred == 'communicated_image_data':
                obj, mode = parts[1], parts[2]

                # Option B: Communicate data already held by a rover
                min_cost_comm_only = math.inf
                for rover, data_obj, data_mode in rover_have_image:
                    if data_obj == obj and data_mode == mode:
                        # This rover has the required image data
                        cost = 0
                        # Cost to communicate from rover's current location
                        min_comm_nav_cost = math.inf
                        if self.comm_waypoints: # Check if there are any communication waypoints
                            min_comm_nav_cost = min(self.get_distance(rover, current_rover_locations[rover], comm_wp) for comm_wp in self.comm_waypoints)

                        if min_comm_nav_cost == math.inf:
                             cost = math.inf # Cannot reach any communication point
                        else:
                             cost += min_comm_nav_cost
                             cost += 1 # communicate action
                        min_cost_comm_only = min(min_cost_comm_only, cost) # Min over rovers holding the data

                min_goal_cost = min(min_goal_cost, min_cost_comm_only)

                # Option A: Get data (image) then communicate
                # Find capable rovers/cameras/modes
                # Iterate through rovers
                for rover in current_rover_locations:
                    if self.rover_equipment.get(rover, {}).get('imaging', False):
                        # Iterate through cameras on this rover
                        for camera, cam_rover in self.on_board.items():
                            if cam_rover == rover and mode in self.supports.get(camera, set()):
                                # Found a capable rover/camera/mode combination
                                # Find image waypoints for this objective
                                image_wps = self.image_waypoints.get(obj, set())
                                if not image_wps:
                                    continue # Cannot image this objective if no visible_from waypoints

                                # Find calibration target and waypoints for this camera
                                cal_target = self.calibration_target.get(camera)
                                cal_wps = self.cal_waypoints.get(camera, set())
                                needs_calibration_target = cal_target is not None

                                # Cost to get image at any image_wp, then communicate from there
                                min_cost_get_then_comm_for_rcm = math.inf # Min cost for this rover/camera/mode over all image_wps
                                for image_wp in image_wps:
                                    cost = 0
                                    # Cost to get image at image_wp
                                    cost_get_image = self.get_distance(rover, current_rover_locations[rover], image_wp) # Nav to image point
                                    cost_get_image += 1 # take_image action

                                    # Calibration cost if needed
                                    cost_calibrate = 0
                                    if (camera, rover) not in camera_calibrated:
                                        # Need calibration
                                        min_cal_nav = math.inf
                                        if needs_calibration_target and cal_wps: # Must have target and cal waypoints
                                            min_cal_nav = min(self.get_distance(rover, image_wp, cal_wp) for cal_wp in cal_wps)
                                        else:
                                             # Cannot calibrate if no target or no cal waypoints
                                             continue # Skip this image_wp for this rover/camera/mode

                                        cost_calibrate = min_cal_nav + 1 # Nav to cal + Calibrate action
                                        # Simplified: ignore nav back cost

                                    cost_get_image += cost_calibrate

                                    # Cost to communicate from image_wp (after taking image)
                                    min_comm_nav = math.inf
                                    if self.comm_waypoints: # Check if there are any communication waypoints
                                        min_comm_nav = min(self.get_distance(rover, image_wp, comm_wp) for comm_wp in self.comm_waypoints)

                                    if min_comm_nav == math.inf:
                                        cost = math.inf # Cannot reach any communication point from image location
                                    else:
                                        cost = cost_get_image + min_comm_nav + 1 # Nav to comm + Communicate action

                                    min_cost_get_then_comm_for_rcm = min(min_cost_get_then_comm_for_rcm, cost) # Min over image_wps

                                min_goal_cost = min(min_goal_cost, min_cost_get_then_comm_for_rcm) # Min over rovers/cameras/modes

            # Add the minimum cost for this goal to the total
            if min_goal_cost == math.inf:
                 # If any goal is impossible from this state, the heuristic is infinity
                 return math.inf

            total_cost += min_goal_cost

        return total_cost

