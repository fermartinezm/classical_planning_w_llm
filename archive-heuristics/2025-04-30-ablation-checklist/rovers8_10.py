# Add necessary imports and base class definition if not provided elsewhere
# from heuristics.heuristic_base import Heuristic # Uncomment this line in the actual environment

# Mock Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

# Add helper functions get_parts and match
import collections
from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle potential empty facts or malformed strings defensively
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS helper function
def bfs(graph, start_node, target_nodes):
    """
    Performs BFS to find the shortest distance from start_node to any node in target_nodes.
    graph: adjacency list {node: {neighbor1, neighbor2, ...}}
    start_node: the starting node
    target_nodes: a set of target nodes
    Returns: shortest distance (number of edges) or float('inf') if no path exists.
    """
    if not target_nodes: # Cannot reach any target if the set is empty
        return float('inf')

    if start_node in target_nodes:
        return 0

    queue = collections.deque([(start_node, 0)])
    visited = {start_node}

    while queue:
        current_node, distance = queue.popleft()

        # Check if current_node is in the graph keys before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor in target_nodes:
                        return distance + 1
                    queue.append((neighbor, distance + 1))

    return float('inf') # No path found to any target node


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing up the estimated costs
    for each unachieved goal predicate. The cost for each predicate is
    estimated based on the first missing step (collecting data/image,
    moving to communication point, communicating), using shortest path
    distances for movement.

    This heuristic is non-admissible and ignores negative effects and
    resource constraints (like store capacity or camera calibration state
    after taking an image) for simplicity and speed.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Collect all objects of relevant types
        self.all_rovers = set()
        self.all_waypoints = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_cameras = set()
        self.all_stores = set()
        self.lander_location = None # Assuming only one lander

        # First pass to collect all objects and lander location
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            if parts[0] == "rover": self.all_rovers.add(parts[1])
            elif parts[0] == "waypoint": self.all_waypoints.add(parts[1])
            elif parts[0] == "objective": self.all_objectives.add(parts[1])
            elif parts[0] == "mode": self.all_modes.add(parts[1])
            elif parts[0] == "camera": self.all_cameras.add(parts[1])
            elif parts[0] == "store": self.all_stores.add(parts[1])
            elif parts[0] == "lander": pass # Ignore lander object type itself
            elif parts[0] == "at_lander":
                 if len(parts) == 3: # Ensure correct number of parts
                    self.lander_location = parts[2] # Capture lander location
                    self.all_waypoints.add(parts[2]) # Lander location is a waypoint

        # Initialize and populate static data structures
        self.rover_capabilities = {rover: {} for rover in self.all_rovers} # rover -> {capability: True}
        self.store_owners = {} # store -> rover
        self.camera_info = {cam: {'supports': set()} for cam in self.all_cameras} # camera -> {on_board: rover, supports: {mode}, cal_target: objective}
        self.visible_from_objective = collections.defaultdict(set) # objective -> {waypoint}
        self.rover_graphs = {rover: collections.defaultdict(set) for rover in self.all_rovers} # rover -> {waypoint: {neighbor}}
        visible_graph = collections.defaultdict(set) # waypoint -> {visible_waypoint} (symmetric)


        # Second pass to populate structures using collected objects
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] in ["equipped_for_soil_analysis", "equipped_for_rock_analysis", "equipped_for_imaging"]:
                if len(parts) == 2 and parts[1] in self.all_rovers:
                    self.rover_capabilities[parts[1]][parts[0]] = True
            elif parts[0] == "store_of":
                if len(parts) == 3 and parts[1] in self.all_stores and parts[2] in self.all_rovers:
                    self.store_owners[parts[1]] = parts[2]
            elif parts[0] == "on_board":
                 if len(parts) == 3 and parts[1] in self.all_cameras and parts[2] in self.all_rovers:
                    self.camera_info[parts[1]]['on_board'] = parts[2]
            elif parts[0] == "supports":
                 if len(parts) == 3 and parts[1] in self.all_cameras and parts[2] in self.all_modes:
                    self.camera_info[parts[1]]['supports'].add(parts[2])
            elif parts[0] == "calibration_target":
                 if len(parts) == 3 and parts[1] in self.all_cameras and parts[2] in self.all_objectives:
                    self.camera_info[parts[1]]['cal_target'] = parts[2]
            elif parts[0] == "visible_from":
                if len(parts) == 3 and parts[1] in self.all_objectives and parts[2] in self.all_waypoints:
                    self.visible_from_objective[parts[1]].add(parts[2])
            elif parts[0] == "can_traverse":
                if len(parts) == 4 and parts[1] in self.all_rovers and parts[2] in self.all_waypoints and parts[3] in self.all_waypoints:
                    rover, wp1, wp2 = parts[1], parts[2], parts[3]
                    self.rover_graphs[rover][wp1].add(wp2)
            elif parts[0] == "visible":
                 if len(parts) == 3 and parts[1] in self.all_waypoints and parts[2] in self.all_waypoints:
                     wp1, wp2 = parts[1], parts[2]
                     visible_graph[wp1].add(wp2)
                     visible_graph[wp2].add(wp1) # Visibility is symmetric

        # Identify communication points (waypoints visible from the lander)
        self.communication_points = set()
        if self.lander_location and self.lander_location in self.all_waypoints:
             self.communication_points.add(self.lander_location) # Lander location is a comm point
             if self.lander_location in visible_graph:
                 self.communication_points.update(visible_graph[self.lander_location])

        # Store goal types for easier lookup
        self.goal_types = collections.defaultdict(set) # predicate -> {args_tuple}
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals
            if parts[0] == "communicated_soil_data":
                 if len(parts) == 2 and parts[1] in self.all_waypoints:
                    self.goal_types[parts[0]].add(tuple(parts[1:]))
            elif parts[0] == "communicated_rock_data":
                 if len(parts) == 2 and parts[1] in self.all_waypoints:
                    self.goal_types[parts[0]].add(tuple(parts[1:]))
            elif parts[0] == "communicated_image_data":
                 if len(parts) == 3 and parts[1] in self.all_objectives and parts[2] in self.all_modes:
                    self.goal_types[parts[0]].add(tuple(parts[1:]))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state facts
        rover_location = {} # rover -> waypoint
        have_soil = collections.defaultdict(dict) # rover -> {waypoint: True}
        have_rock = collections.defaultdict(dict) # rover -> {waypoint: True}
        have_image = collections.defaultdict(lambda: collections.defaultdict(dict)) # rover -> {objective: {mode: True}}
        # at_soil = set() # {waypoint} # Not strictly needed for this simplified heuristic
        # at_rock = set() # {waypoint} # Not strictly needed for this simplified heuristic
        # calibrated_camera = collections.defaultdict(dict) # camera -> {rover: True} # Ignored in simplified image cost
        communicated_soil = set() # {waypoint}
        communicated_rock = set() # {waypoint}
        communicated_image = collections.defaultdict(set) # objective -> {mode}
        # store_empty = set() # {store} # Ignored
        # store_full = set() # {store} # Ignored

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == "at":
                if len(parts) == 3 and parts[1] in self.all_rovers and parts[2] in self.all_waypoints:
                    rover_location[parts[1]] = parts[2]
            elif parts[0] == "have_soil_analysis":
                if len(parts) == 3 and parts[1] in self.all_rovers and parts[2] in self.all_waypoints:
                    have_soil[parts[1]][parts[2]] = True
            elif parts[0] == "have_rock_analysis":
                if len(parts) == 3 and parts[1] in self.all_rovers and parts[2] in self.all_waypoints:
                    have_rock[parts[1]][parts[2]] = True
            elif parts[0] == "have_image":
                 if len(parts) == 4 and parts[1] in self.all_rovers and parts[2] in self.all_objectives and parts[3] in self.all_modes:
                    have_image[parts[1]][parts[2]][parts[3]] = True
            # elif parts[0] == "at_soil_sample": at_soil.add(parts[1]) # Relaxed
            # elif parts[0] == "at_rock_sample": at_rock.add(parts[1]) # Relaxed
            # elif parts[0] == "calibrated": calibrated_camera[parts[1]][parts[2]] = True # Relaxed
            elif parts[0] == "communicated_soil_data":
                if len(parts) == 2 and parts[1] in self.all_waypoints:
                    communicated_soil.add(parts[1])
            elif parts[0] == "communicated_rock_data":
                if len(parts) == 2 and parts[1] in self.all_waypoints:
                    communicated_rock.add(parts[1])
            elif parts[0] == "communicated_image_data":
                if len(parts) == 3 and parts[1] in self.all_objectives and parts[2] in self.all_modes:
                    communicated_image[parts[1]].add(parts[2])
            # elif parts[0] == "empty": store_empty.add(parts[1]) # Relaxed
            # elif parts[0] == "full": store_full.add(parts[1]) # Relaxed


        total_cost = 0

        # Process unachieved soil goals
        for waypoint_goal in self.goal_types.get("communicated_soil_data", set()):
            waypoint = waypoint_goal[0]
            if waypoint not in communicated_soil:
                min_goal_cost = float('inf')

                # Case 1: Data already collected by some rover
                rover_with_data = None
                for rover in self.all_rovers:
                    if have_soil.get(rover, {}).get(waypoint):
                        rover_with_data = rover
                        break # Found a rover with the data

                if rover_with_data:
                    # Cost = Communicate (1) + Move to comm point
                    loc_r = rover_location.get(rover_with_data)
                    if loc_r: # Rover must have a location
                         dist_to_comm = bfs(self.rover_graphs.get(rover_with_data, {}), loc_r, self.communication_points)
                         if dist_to_comm != float('inf'):
                             min_goal_cost = min(min_goal_cost, 1 + dist_to_comm) # 1 for communicate

                # Case 2: Data needs collecting (Assume sample is available if goal not met and data not held)
                if min_goal_cost == float('inf'): # Only consider collecting if data isn't held
                    for rover in self.all_rovers:
                        if self.rover_capabilities.get(rover, {}).get("equipped_for_soil_analysis"):
                            # Cost = Move to sample + Sample + Move to comm + Communicate
                            loc_r = rover_location.get(rover)
                            if loc_r: # Rover must have a location
                                dist_to_sample = bfs(self.rover_graphs.get(rover, {}), loc_r, {waypoint})
                                if dist_to_sample != float('inf'):
                                    dist_sample_to_comm = bfs(self.rover_graphs.get(rover, {}), waypoint, self.communication_points)
                                    if dist_sample_to_comm != float('inf'):
                                         # Sample (1) + Communicate (1) = 2 actions + movement
                                         current_rover_cost = dist_to_sample + 1 + dist_sample_to_comm + 1
                                         min_goal_cost = min(min_goal_cost, current_rover_cost)

                total_cost += min_goal_cost if min_goal_cost != float('inf') else 1000 # Add a large penalty if impossible

        # Process unachieved rock goals
        for waypoint_goal in self.goal_types.get("communicated_rock_data", set()):
            waypoint = waypoint_goal[0]
            if waypoint not in communicated_rock:
                min_goal_cost = float('inf')

                # Case 1: Data already collected by some rover
                rover_with_data = None
                for rover in self.all_rovers:
                    if have_rock.get(rover, {}).get(waypoint):
                        rover_with_data = rover
                        break # Found a rover with the data

                if rover_with_data:
                    # Cost = Communicate (1) + Move to comm point
                    loc_r = rover_location.get(rover_with_data)
                    if loc_r: # Rover must have a location
                         dist_to_comm = bfs(self.rover_graphs.get(rover_with_data, {}), loc_r, self.communication_points)
                         if dist_to_comm != float('inf'):
                             min_goal_cost = min(min_goal_cost, 1 + dist_to_comm) # 1 for communicate

                # Case 2: Data needs collecting (Assume sample is available if goal not met and data not held)
                if min_goal_cost == float('inf'): # Only consider collecting if data isn't held
                    for rover in self.all_rovers:
                        if self.rover_capabilities.get(rover, {}).get("equipped_for_rock_analysis"):
                            # Cost = Move to sample + Sample + Move to comm + Communicate
                            loc_r = rover_location.get(rover)
                            if loc_r: # Rover must have a location
                                dist_to_sample = bfs(self.rover_graphs.get(rover, {}), loc_r, {waypoint})
                                if dist_to_sample != float('inf'):
                                    dist_sample_to_comm = bfs(self.rover_graphs.get(rover, {}), waypoint, self.communication_points)
                                    if dist_sample_to_comm != float('inf'):
                                         # Sample (1) + Communicate (1) = 2 actions + movement
                                         current_rover_cost = dist_to_sample + 1 + dist_sample_to_comm + 1
                                         min_goal_cost = min(min_goal_cost, current_rover_cost)

                total_cost += min_goal_cost if min_goal_cost != float('inf') else 1000 # Add a large penalty if impossible


        # Process unachieved image goals
        for obj_mode_goal in self.goal_types.get("communicated_image_data", set()):
            objective, mode = obj_mode_goal
            if mode not in communicated_image.get(objective, set()):
                min_goal_cost = float('inf')

                # Case 1: Image already taken by some rover
                rover_with_image = None
                for rover in self.all_rovers:
                    if have_image.get(rover, {}).get(objective, {}).get(mode):
                        rover_with_image = rover
                        break # Found a rover with the image

                if rover_with_image:
                    # Cost = Communicate (1) + Move to comm point
                    loc_r = rover_location.get(rover_with_image)
                    if loc_r: # Rover must have a location
                         dist_to_comm = bfs(self.rover_graphs.get(rover_with_image, {}), loc_r, self.communication_points)
                         if dist_to_comm != float('inf'):
                             min_goal_cost = min(min_goal_cost, 1 + dist_to_comm) # 1 for communicate

                # Case 2: Image needs taking (Relaxed cost)
                if min_goal_cost == float('inf'): # Only consider taking if image isn't held
                    # Find suitable rover/camera combinations
                    for rover in self.all_rovers:
                        if self.rover_capabilities.get(rover, {}).get("equipped_for_imaging"):
                            for camera in self.all_cameras:
                                # Check if camera is on this rover and supports the mode
                                if self.camera_info.get(camera, {}).get('on_board') == rover and \
                                   mode in self.camera_info.get(camera, {}).get('supports', set()):
                                    # Find waypoints visible from the objective
                                    possible_image_wps = self.visible_from_objective.get(objective, set())
                                    if possible_image_wps:
                                        # Find the best image waypoint P to go to
                                        loc_r = rover_location.get(rover)
                                        if loc_r: # Rover must have a location
                                            min_dist_to_img_wp = float('inf')
                                            best_img_wp = None
                                            for img_wp in possible_image_wps:
                                                dist = bfs(self.rover_graphs.get(rover, {}), loc_r, {img_wp})
                                                if dist < min_dist_to_img_wp:
                                                    min_dist_to_img_wp = dist
                                                    best_img_wp = img_wp

                                            if best_img_wp and min_dist_to_img_wp != float('inf'):
                                                # Cost = Move to ImageWP + Calibrate+TakeImage (2) + Move from ImageWP to Comm + Communicate (1)
                                                dist_img_wp_to_comm = bfs(self.rover_graphs.get(rover, {}), best_img_wp, self.communication_points)
                                                if dist_img_wp_to_comm != float('inf'):
                                                    current_rover_cost = min_dist_to_img_wp + 2 + dist_img_wp_to_comm + 1
                                                    min_goal_cost = min(min_goal_cost, current_rover_cost)


                total_cost += min_goal_cost if min_goal_cost != float('inf') else 1000 # Add a large penalty if impossible


        return total_cost
