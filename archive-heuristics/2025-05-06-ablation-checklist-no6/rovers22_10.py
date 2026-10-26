from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# Helper function for BFS
def bfs(graph, start_node):
    """Computes shortest distances from start_node to all other nodes in the graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node might not be in the graph if it's an object type not a waypoint
         # Or if the graph is empty
         return distances # Return distances with all infinity

    distances[start_node] = 0
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        for neighbor in graph.get(current_node, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the number of actions required to satisfy unsatisfied goal conditions.
    The heuristic sums the estimated costs for each unsatisfied goal, considering
    sampling/imaging, calibration (for images), communication, and necessary movements.

    Movement costs are estimated using precomputed shortest paths on the waypoint graph.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Precomputes waypoint distances and relevant static information.
        """
        self.goals = task.goals

        # --- Precompute static information ---
        self.rover_capabilities = {} # {rover: {'soil': bool, 'rock': bool, 'imaging': bool}}
        self.rover_stores = {}       # {rover: store} # {rover_name: store_name}
        self.rover_cameras = {}      # {rover: {camera: {'modes': [mode], 'target': obj}}}
        self.obj_visible_wps = {}    # {objective: [waypoint]}
        self.caltarget_visible_wps = {} # {objective: [waypoint]} # Note: Calibration targets are objectives
        self.lander_wp = None
        self.waypoints = set()
        self.graph = {} # Waypoint graph based on visible predicate
        self.dist = {}  # Shortest distances between waypoints

        # Collect all objects first to handle dependencies
        all_objects = set()
        # Collect objects from initial state
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts: all_objects.update(parts[1:]) # Add all arguments as potential objects
        # Collect objects from static facts
        for fact in task.static:
             parts = get_parts(fact)
             if parts: all_objects.update(parts[1:]) # Add all arguments as potential objects
        # Collect objects from goals
        for goal in task.goals:
             parts = get_parts(goal)
             if parts: all_objects.update(parts[1:]) # Add all arguments as potential objects

        # Initialize structures for all potential rovers/cameras/objectives found
        for obj in all_objects:
            if obj.startswith('rover'):
                 self.rover_capabilities[obj] = {'soil': False, 'rock': False, 'imaging': False}
                 self.rover_cameras[obj] = {}
            elif obj.startswith('objective'):
                 self.obj_visible_wps[obj] = []
                 self.caltarget_visible_wps[obj] = [] # Objectives can be calibration targets
            elif obj.startswith('waypoint'):
                 self.waypoints.add(obj)


        # Parse static facts
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at_lander':
                # Assuming only one lander and its location is static
                if len(args) == 2:
                    self.lander_wp = args[1]
                    self.waypoints.add(args[1])
            elif predicate == 'equipped_for_soil_analysis':
                if len(args) == 1 and args[0] in self.rover_capabilities:
                    self.rover_capabilities[args[0]]['soil'] = True
            elif predicate == 'equipped_for_rock_analysis':
                 if len(args) == 1 and args[0] in self.rover_capabilities:
                    self.rover_capabilities[args[0]]['rock'] = True
            elif predicate == 'equipped_for_imaging':
                 if len(args) == 1 and args[0] in self.rover_capabilities:
                    self.rover_capabilities[args[0]]['imaging'] = True
            elif predicate == 'store_of':
                 # args[0] is store, args[1] is rover
                 if len(args) == 2 and args[1] in self.rover_capabilities:
                    self.rover_stores[args[1]] = args[0]
            elif predicate == 'visible':
                if len(args) == 2:
                    wp1, wp2 = args
                    self.waypoints.add(wp1)
                    self.waypoints.add(wp2)
                    self.graph.setdefault(wp1, set()).add(wp2)
                    self.graph.setdefault(wp2, set()).add(wp1) # Assuming symmetric visibility
            # elif predicate == 'can_traverse': # Ignoring can_traverse for general waypoint graph
            #     pass # Can traverse is usually implied by visible for rovers in this domain
            elif predicate == 'on_board':
                if len(args) == 2 and args[1] in self.rover_cameras:
                    camera, rover = args
                    self.rover_cameras[rover].setdefault(camera, {'modes': [], 'target': None})
            elif predicate == 'supports':
                if len(args) == 2:
                    camera, mode = args
                    # Find which rover this camera is on
                    rover_with_camera = None
                    for r, cameras in self.rover_cameras.items():
                        if camera in cameras:
                            rover_with_camera = r
                            break
                    if rover_with_camera:
                        self.rover_cameras[rover_with_camera][camera]['modes'].append(mode)
            elif predicate == 'calibration_target':
                if len(args) == 2:
                    camera, target_obj = args
                     # Find which rover this camera is on
                    rover_with_camera = None
                    for r, cameras in self.rover_cameras.items():
                        if camera in cameras:
                            rover_with_camera = r
                            break
                    if rover_with_camera:
                        self.rover_cameras[rover_with_camera][camera]['target'] = target_obj
                        # Ensure target obj is in caltarget_visible_wps if it's a calibration target
                        self.caltarget_visible_wps.setdefault(target_obj, [])
            elif predicate == 'visible_from':
                if len(args) == 2:
                    obj, wp = args
                    if obj in self.obj_visible_wps:
                        self.obj_visible_wps[obj].append(wp)
                    # Check if this objective is a calibration target
                    is_cal_target = False
                    for r, cameras in self.rover_cameras.items():
                        for c, info in cameras.items():
                            if info.get('target') == obj:
                                is_cal_target = True
                                break
                        if is_cal_target: break
                    if is_cal_target:
                         if obj in self.caltarget_visible_wps:
                            self.caltarget_visible_wps[obj].append(wp)

        # Compute all-pairs shortest paths
        # Ensure all waypoints mentioned in static/initial/goals are in the graph nodes
        for wp in self.waypoints:
             if wp not in self.graph:
                  self.graph[wp] = set() # Add isolated waypoints to graph structure

        for start_wp in self.graph.keys(): # Iterate over actual nodes in graph
            self.dist[start_wp] = bfs(self.graph, start_wp)

        # Find waypoints visible from the lander
        self.lander_visible_wps = []
        if self.lander_wp and self.graph.get(self.lander_wp):
             self.lander_visible_wps = list(self.graph[self.lander_wp])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse current state ---
        current_at = {} # {rover: waypoint}
        current_have_soil = {} # {(rover, wp): True}
        current_have_rock = {} # {(rover, wp): True}
        current_have_image = {} # {(rover, obj, mode): True}
        current_full_stores = set() # {store}
        current_calibrated_cameras = set() # {(camera, rover)}
        current_communicated_soil = set() # {wp}
        current_communicated_rock = set() # {wp}
        current_communicated_images = set() # {(obj, mode)}
        current_at_soil_sample = set() # {wp}
        current_at_rock_sample = set() # {wp}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at':
                if len(args) == 2: current_at[args[0]] = args[1]
            elif predicate == 'have_soil_analysis':
                 if len(args) == 2: current_have_soil[(args[0], args[1])] = True
            elif predicate == 'have_rock_analysis':
                 if len(args) == 2: current_have_rock[(args[0], args[1])] = True
            elif predicate == 'have_image':
                 if len(args) == 3: current_have_image[(args[0], args[1], args[2])] = True
            elif predicate == 'full':
                 if len(args) == 1: current_full_stores.add(args[0])
            elif predicate == 'calibrated':
                 if len(args) == 2: current_calibrated_cameras.add((args[0], args[1]))
            elif predicate == 'communicated_soil_data':
                 if len(args) == 1: current_communicated_soil.add(args[0])
            elif predicate == 'communicated_rock_data':
                 if len(args) == 1: current_communicated_rock.add(args[0])
            elif predicate == 'communicated_image_data':
                 if len(args) == 2: current_communicated_images.add((args[0], args[1]))
            elif predicate == 'at_soil_sample':
                 if len(args) == 1: current_at_soil_sample.add(args[0])
            elif predicate == 'at_rock_sample':
                 if len(args) == 1: current_at_rock_sample.add(args[0])


        # --- Compute heuristic value ---
        h = 0

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'communicated_soil_data':
                if len(args) != 1: continue
                wp = args[0]
                if wp not in current_communicated_soil:
                    goal_h = float('inf')
                    for rover, caps in self.rover_capabilities.items():
                        if caps.get('soil', False): # Check if capable
                            rover_h = 0
                            has_sample = (rover, wp) in current_have_soil

                            if not has_sample:
                                if wp not in current_at_soil_sample:
                                    # Cannot sample here, this rover cannot achieve this goal from scratch
                                    continue

                                rover_h += 1 # sample action
                                rover_current_pos = current_at.get(rover)
                                if rover_current_pos is None:
                                     # Rover location unknown, cannot estimate movement
                                     continue
                                dist_to_sample = self.dist.get(rover_current_pos, {}).get(wp, float('inf'))
                                if dist_to_sample == float('inf'):
                                     # Cannot reach sample location
                                     continue
                                rover_h += dist_to_sample

                                store = self.rover_stores.get(rover)
                                if store and store in current_full_stores:
                                    rover_h += 1 # drop action

                            # Need to communicate
                            rover_h += 1 # communicate action
                            min_comm_dist = float('inf')
                            rover_current_pos = current_at.get(rover) # Re-get pos in case of movement (though heuristic doesn't change state)
                            if rover_current_pos is None: continue # Should not happen if we got here
                            if not self.lander_visible_wps:
                                # No communication points available
                                min_comm_dist = float('inf')
                            else:
                                min_comm_dist = min((self.dist.get(rover_current_pos, {}).get(comm_wp, float('inf')) for comm_wp in self.lander_visible_wps), default=float('inf'))


                            if min_comm_dist == float('inf'):
                                # Cannot reach a communication point
                                continue

                            rover_h += min_comm_dist
                            goal_h = min(goal_h, rover_h)

                    if goal_h == float('inf'):
                         # This goal is impossible from this state
                         return float('inf')
                    h += goal_h

            elif predicate == 'communicated_rock_data':
                if len(args) != 1: continue
                wp = args[0]
                if wp not in current_communicated_rock:
                    goal_h = float('inf')
                    for rover, caps in self.rover_capabilities.items():
                        if caps.get('rock', False): # Check if capable
                            rover_h = 0
                            has_sample = (rover, wp) in current_have_rock

                            if not has_sample:
                                if wp not in current_at_rock_sample:
                                     # Cannot sample here
                                     continue

                                rover_h += 1 # sample action
                                rover_current_pos = current_at.get(rover)
                                if rover_current_pos is None: continue
                                dist_to_sample = self.dist.get(rover_current_pos, {}).get(wp, float('inf'))
                                if dist_to_sample == float('inf'): continue
                                rover_h += dist_to_sample

                                store = self.rover_stores.get(rover)
                                if store and store in current_full_stores:
                                    rover_h += 1 # drop action

                            # Need to communicate
                            rover_h += 1 # communicate action
                            min_comm_dist = float('inf')
                            rover_current_pos = current_at.get(rover)
                            if rover_current_pos is None: continue
                            if not self.lander_visible_wps:
                                min_comm_dist = float('inf')
                            else:
                                min_comm_dist = min((self.dist.get(rover_current_pos, {}).get(comm_wp, float('inf')) for comm_wp in self.lander_visible_wps), default=float('inf'))

                            if min_comm_dist == float('inf'): continue

                            rover_h += min_comm_dist
                            goal_h = min(goal_h, rover_h)

                    if goal_h == float('inf'): return float('inf')
                    h += goal_h

            elif predicate == 'communicated_image_data':
                if len(args) != 2: continue
                o, m = args
                if (o, m) not in current_communicated_images:
                    goal_h = float('inf')
                    for rover, caps in self.rover_capabilities.items():
                        if caps.get('imaging', False): # Check if capable
                            for camera, cam_info in self.rover_cameras.get(rover, {}).items():
                                if m in cam_info.get('modes', []): # Check if mode supported
                                    rover_camera_h = 0
                                    has_image = (rover, o, m) in current_have_image

                                    if not has_image:
                                        rover_camera_h += 1 # take_image action

                                        # Need to move to image waypoint
                                        img_wps = self.obj_visible_wps.get(o, [])
                                        if not img_wps:
                                            # Cannot image this objective
                                            rover_camera_h = float('inf')
                                            continue
                                        min_img_dist = float('inf')
                                        rover_current_pos = current_at.get(rover)
                                        if rover_current_pos is None: continue
                                        min_img_dist = min((self.dist.get(rover_current_pos, {}).get(p, float('inf')) for p in img_wps), default=float('inf'))
                                        if min_img_dist == float('inf'):
                                             # Cannot reach any image waypoint
                                             rover_camera_h = float('inf')
                                             continue
                                        # Add movement cost to image waypoint
                                        rover_camera_h += min_img_dist

                                        # Need to calibrate (assuming needed before taking image if image is not held)
                                        cal_target = cam_info.get('target')
                                        if cal_target is None:
                                             # Camera has no calibration target defined
                                             rover_camera_h = float('inf')
                                             continue

                                        cal_wps = self.caltarget_visible_wps.get(cal_target, [])
                                        if not cal_wps:
                                            # Cannot calibrate this camera (no visible waypoint for target)
                                            rover_camera_h = float('inf')
                                            continue
                                        min_cal_dist = float('inf')
                                        # Use current pos for simplicity, not pos after moving to img_wp
                                        # This underestimates movement if calibration and imaging are at different WPs
                                        # Let's use current pos -> cal_wp dist as a simple estimate
                                        if rover_current_pos is None: continue
                                        min_cal_dist = min((self.dist.get(rover_current_pos, {}).get(w, float('inf')) for w in cal_wps), default=float('inf'))
                                        if min_cal_dist == float('inf'):
                                             # Cannot reach any calibration waypoint
                                             rover_camera_h = float('inf')
                                             continue

                                        rover_camera_h += 1 # calibrate action
                                        # Add movement cost to calibration waypoint
                                        rover_camera_h += min_cal_dist


                                    # Need to communicate
                                    rover_camera_h += 1 # communicate action
                                    min_comm_dist = float('inf')
                                    rover_current_pos = current_at.get(rover)
                                    if rover_current_pos is None: continue
                                    if not self.lander_visible_wps:
                                        min_comm_dist = float('inf')
                                    else:
                                        min_comm_dist = min((self.dist.get(rover_current_pos, {}).get(comm_wp, float('inf')) for comm_wp in self.lander_visible_wps), default=float('inf'))

                                    if min_comm_dist == float('inf'):
                                         # Cannot reach a communication point
                                         continue

                                    rover_camera_h += min_comm_dist

                                    goal_h = min(goal_h, rover_camera_h)

                    if goal_h == float('inf'):
                         # This goal is impossible from this state
                         return float('inf')
                    h += goal_h

        return h
