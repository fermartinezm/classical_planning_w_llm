from fnmatch import fnmatch
from collections import deque
import math

# Assume Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Perform BFS to find shortest distances from start_node to all reachable nodes.
    Returns a dictionary {node: distance}.
    """
    # Handle empty graph or start_node not in graph
    if not graph or start_node not in graph:
        # If start_node is not in graph but graph is not empty, it's isolated.
        # If graph is empty, no nodes exist.
        # In either case, only the start_node itself is reachable with distance 0 if it exists.
        return {start_node: 0} if start_node in graph else {}

    distances = {node: math.inf for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        # Iterate over neighbors using .get for safety, although graph keys should be comprehensive
        for neighbor in graph.get(current_node, []):
            # Check if neighbor exists in distances (should be true if graph keys are comprehensive)
            # and if it hasn't been visited yet (distance is infinity)
            if distances.get(neighbor, math.inf) == math.inf:
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    goal conditions, focusing on the steps needed to collect, analyze (if applicable),
    and communicate data (soil, rock, image). It includes navigation costs
    estimated by shortest paths between relevant waypoints.

    # Assumptions
    - The primary goals are communicating data (soil, rock, image).
    - Samples at waypoints are inexhaustible resources (based on PDDL effect analysis).
    - Lander location is static.
    - Rover traversal capabilities are static and the same for all rovers (based on example instances).
    - Camera properties (on_board, supports, calibration_target) are static.
    - Objective visibility from waypoints is static.
    - Each action (navigate, take, analyse, calibrate, take_image, communicate, drop) costs 1.
    - Dropping a sample takes 1 action and can be done at the current location.

    # Heuristic Initialization
    - Parses static facts and initial state to build data structures for:
        - Lander location.
        - Rover capabilities (soil, rock, imaging).
        - Store ownership by rovers.
        - Camera properties (on_board, supported modes, calibration target).
        - Objective visibility from waypoints.
        - Waypoint graph based on 'can_traverse' (assuming it's uniform across rovers).
        - Locations of soil and rock samples.
    - Identifies all unique waypoints involved in the problem.
    - Computes all-pairs shortest paths between waypoints using BFS on the traversal graph.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is the sum of estimated costs for each unsatisfied goal.
    For each unsatisfied goal (communicated_soil_data, communicated_rock_data, communicated_image_data):

    1.  Identify the type of data required (soil, rock, or image) and the specific item (waypoint W or objective O/mode M).
    2.  Find all rovers capable of handling this data type (equipped_for_...).
    3.  For each capable rover, calculate the minimum estimated cost to achieve *this specific goal* using that rover, considering its current state and location. This involves estimating the steps and navigation required based on the most advanced state achieved for this goal by this rover:
        *   **For Soil/Rock Data (waypoint W):**
            *   If the rover already has the analysed data for W: Cost = Navigate(rover_loc, lander_loc) + Communicate.
            *   If the rover has the sample data for W but not analysed: Cost = Navigate(rover_loc, W) + Analyse + Navigate(W, lander_loc) + Communicate.
            *   If the sample exists at W but the rover doesn't have the data: Cost = Navigate(rover_loc, W) + TakeSample + Analyse + Navigate(W, lander_loc) + Communicate. (Requires empty store; add DropSample cost if store is full).
            *   If the sample doesn't exist at W (static check): This path is impossible for this goal via this waypoint.
        *   **For Image Data (objective O, mode M):**
            *   Find cameras on the rover supporting mode M.
            *   For each suitable camera C:
                *   If the rover already has the image (O, M): Cost = Navigate(rover_loc, lander_loc) + Communicate.
                *   If the camera C is calibrated on this rover: Cost = Navigate(rover_loc, W_img) + TakeImage + Navigate(W_img, lander_loc) + Communicate, minimizing total navigation over W_img visible from O.
                *   If the camera C is not calibrated: Cost = Navigate(rover_loc, W_cal) + Calibrate + Navigate(W_cal, W_img) + TakeImage + Navigate(W_img, lander_loc) + Communicate, minimizing total navigation over W_cal visible from C's calibration target and W_img visible from O.
                *   If required visible_from waypoints don't exist (static check): This path is impossible.
    4.  Take the minimum cost calculated across all suitable rovers (and cameras for images) for this specific goal.
    5.  If no path is found (e.g., required waypoint unreachable, no equipped rover), assign a large penalty cost (e.g., 1000) for this goal.
    6.  Sum the minimum costs for all unsatisfied goals.
    """

    def __init__(self, task):
        super().__init__(task)

        # --- Precompute Static Information ---

        self.lander_location = None
        self.rover_capabilities = {} # {rover_name: {capability1, capability2, ...}}
        self.store_to_rover = {}     # {store_name: rover_name}
        self.camera_info = {}        # {camera_name: {'on_board': rover, 'supports': {mode1, ...}, 'calibration_target': objective}}
        self.objective_visibility = {} # {objective_name: {waypoint1, waypoint2, ...}}
        self.waypoint_graph = {}     # {waypoint_name: {neighbor1, neighbor2, ...}}
        self.at_sample_locations = {'soil': set(), 'rock': set()} # {type: {waypoint1, ...}}

        all_waypoints_set = set() # Collect all waypoints mentioned

        # Parse static facts
        for fact in self.static:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == 'at_lander':
                self.lander_location = parts[2]
                all_waypoints_set.add(self.lander_location)
            elif predicate.startswith('equipped_for_'):
                capability = predicate # e.g., 'equipped_for_soil_analysis'
                rover = parts[1]
                if rover not in self.rover_capabilities:
                    self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add(capability)
            elif predicate == 'store_of':
                store, rover = parts[1], parts[2]
                self.store_to_rover[store] = rover
            elif predicate == 'on_board':
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['on_board'] = rover
            elif predicate == 'supports':
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                if 'supports' not in self.camera_info[camera]: self.camera_info[camera]['supports'] = set()
                self.camera_info[camera]['supports'].add(mode)
            elif predicate == 'calibration_target':
                camera, objective = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['calibration_target'] = objective
            elif predicate == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                if objective not in self.objective_visibility: self.objective_visibility[objective] = set()
                self.objective_visibility[objective].add(waypoint)
                all_waypoints_set.add(waypoint)
            elif predicate == 'can_traverse':
                 # Assuming can_traverse is the same for all rovers for graph building
                 wp1, wp2 = parts[2], parts[3]
                 if wp1 not in self.waypoint_graph: self.waypoint_graph[wp1] = set()
                 if wp2 not in self.waypoint_graph: self.waypoint_graph[wp2] = set()
                 self.waypoint_graph[wp1].add(wp2)
                 self.waypoint_graph[wp2].add(wp1) # Assuming bidirectional traversal
                 all_waypoints_set.add(wp1)
                 all_waypoints_set.add(wp2)

        # Parse initial state for sample locations (which are static in this domain)
        for fact in self.task.initial_state:
             parts = get_parts(fact)
             predicate = parts[0]
             if predicate == 'at_rock_sample':
                 self.at_sample_locations['rock'].add(parts[1])
                 all_waypoints_set.add(parts[1])
             elif predicate == 'at_soil_sample':
                 self.at_sample_locations['soil'].add(parts[1])
                 all_waypoints_set.add(parts[1])
             elif predicate == 'at': # Add initial rover locations to waypoints
                 obj_name, loc_name = parts[1], parts[2]
                 # Assuming only rovers are 'at' waypoints in the state we care about for location
                 if obj_name in self.rover_capabilities: # Check if it's a rover
                     all_waypoints_set.add(loc_name)


        # Ensure all collected waypoints are in the graph keys for BFS
        for wp in all_waypoints_set:
            if wp not in self.waypoint_graph:
                self.waypoint_graph[wp] = set() # Add isolated waypoints

        all_waypoints = list(self.waypoint_graph.keys()) # Use list for consistent iteration

        # Compute all-pairs shortest paths
        self.distances = {} # {(wp1, wp2): distance}
        for start_wp in all_waypoints:
            distances_from_start = bfs(self.waypoint_graph, start_wp)
            for end_wp, dist in distances_from_start.items():
                self.distances[(start_wp, end_wp)] = dist

    def get_distance(self, wp1, wp2):
        """Looks up precomputed distance, returns infinity if no path."""
        if wp1 == wp2:
            return 0
        return self.distances.get((wp1, wp2), math.inf)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0
        infinity = math.inf # Use math.inf for clarity
        IMPOSSIBLE_GOAL_PENALTY = 1000 # Penalty for goals that seem impossible

        # --- Parse Current State ---
        current_rover_locations = {} # {rover_name: waypoint}
        current_store_status = {}    # {store_name: 'empty' or 'full'}
        current_have_soil = set()    # {(rover, waypoint)}
        current_have_rock = set()    # {(rover, waypoint)}
        current_have_image = set()   # {(rover, objective, mode)}
        current_analysed_soil = set() # {(rover, waypoint)}
        current_analysed_rock = set() # {(rover, waypoint)}
        current_calibrated = set()   # {(camera, rover)}
        current_communicated_soil = set() # {waypoint}
        current_communicated_rock = set() # {waypoint}
        current_communicated_image = set() # {(objective, mode)}

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at':
                obj_name, loc_name = parts[1], parts[2]
                # Assuming only rovers are 'at' waypoints in the state we care about for location
                if obj_name in self.rover_capabilities: # Check if it's a rover
                     current_rover_locations[obj_name] = loc_name
            elif predicate in ['empty', 'full']:
                 store_name = parts[1]
                 current_store_status[store_name] = predicate
            elif predicate == 'have_soil_analysis':
                 current_have_soil.add((parts[1], parts[2]))
            elif predicate == 'have_rock_analysis':
                 current_have_rock.add((parts[1], parts[2]))
            elif predicate == 'have_image':
                 current_have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == 'analysed_soil':
                 current_analysed_soil.add((parts[1], parts[2]))
            elif predicate == 'analysed_rock':
                 current_analysed_rock.add((parts[1], parts[2]))
            elif predicate == 'calibrated':
                 current_calibrated.add((parts[1], parts[2]))
            elif predicate == 'communicated_soil_data':
                 current_communicated_soil.add(parts[1])
            elif predicate == 'communicated_rock_data':
                 current_communicated_rock.add(parts[1])
            elif predicate == 'communicated_image_data':
                 current_communicated_image.add((parts[1], parts[2]))

        # --- Calculate Cost for Unsatisfied Goals ---

        # Group goals by type
        soil_goals = set()
        rock_goals = set()
        image_goals = set()

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'communicated_soil_data':
                soil_goals.add(parts[1])
            elif predicate == 'communicated_rock_data':
                rock_goals.add(parts[1])
            elif predicate == 'communicated_image_data':
                image_goals.add((parts[1], parts[2]))

        # Cost for Soil Goals
        for w in soil_goals:
            if w in current_communicated_soil:
                continue # Goal already met

            min_goal_cost = infinity

            # Check all soil-equipped rovers
            for rover, capabilities in self.rover_capabilities.items():
                if 'equipped_for_soil_analysis' not in capabilities:
                    continue

                p_rover = current_rover_locations.get(rover)
                if p_rover is None: continue # Rover location unknown

                rover_stores = [s for s, r in self.store_to_rover.items() if r == rover]
                if not rover_stores: continue # Rover has no store
                store = rover_stores[0] # Assuming one store per rover

                current_rover_goal_cost = infinity

                # Cost calculation based on the most advanced state achieved by this rover for this goal
                cost_option1 = infinity # Analysed -> Communicate
                cost_option2 = infinity # Have Sample -> Analyse -> Communicate
                cost_option3 = infinity # At Sample -> Take Sample -> Analyse -> Communicate

                # Option 1: Already analysed
                if (rover, w) in current_analysed_soil:
                    nav_cost = self.get_distance(p_rover, self.lander_location)
                    if nav_cost != infinity:
                        cost_option1 = nav_cost + 1 # Nav + Communicate

                # Option 2: Have sample, need analyse
                if (rover, w) in current_have_soil:
                     # Need to be at waypoint w to analyse
                     nav_cost1 = self.get_distance(p_rover, w) # Nav to w
                     nav_cost2 = self.get_distance(w, self.lander_location) # Nav from w to lander
                     if nav_cost1 != infinity and nav_cost2 != infinity:
                         cost_option2 = nav_cost1 + 1 + nav_cost2 + 1 # Nav1 + Analyse + Nav2 + Communicate

                # Option 3: Sample exists, need take sample + analyse
                if w in self.at_sample_locations['soil']:
                    # Need empty store. If not empty, need to drop first.
                    drop_cost = 0
                    if current_store_status.get(store) == 'full':
                        drop_cost = 1 # Drop at current location

                    # Need to be at waypoint w to take sample
                    nav_cost1 = self.get_distance(p_rover, w) # Nav to w
                    nav_cost2 = self.get_distance(w, self.lander_location) # Nav from w to lander

                    if nav_cost1 != infinity and nav_cost2 != infinity:
                         cost_option3 = drop_cost + nav_cost1 + 1 # Drop (if needed) + Nav to w + Take Sample
                         cost_option3 += 1 # Analyse
                         cost_option3 += nav_cost2 + 1 # Nav from w + Communicate


                current_rover_goal_cost = min(cost_option1, cost_option2, cost_option3)
                min_goal_cost = min(min_goal_cost, current_rover_goal_cost)


            # Add cost for this goal
            if min_goal_cost == infinity:
                 total_cost += IMPOSSIBLE_GOAL_PENALTY # Large penalty for seemingly impossible goals
            else:
                 total_cost += min_goal_cost


        # Cost for Rock Goals (similar to Soil)
        for w in rock_goals:
            if w in current_communicated_rock:
                continue # Goal already met

            min_goal_cost = infinity

            # Check all rock-equipped rovers
            for rover, capabilities in self.rover_capabilities.items():
                if 'equipped_for_rock_analysis' not in capabilities:
                    continue

                p_rover = current_rover_locations.get(rover)
                if p_rover is None: continue

                rover_stores = [s for s, r in self.store_to_rover.items() if r == rover]
                if not rover_stores: continue
                store = rover_stores[0]

                current_rover_goal_cost = infinity

                cost_option1 = infinity # Analysed -> Communicate
                cost_option2 = infinity # Have Sample -> Analyse -> Communicate
                cost_option3 = infinity # At Sample -> Take Sample -> Analyse -> Communicate

                # Option 1: Already analysed
                if (rover, w) in current_analysed_rock:
                    nav_cost = self.get_distance(p_rover, self.lander_location)
                    if nav_cost != infinity:
                        cost_option1 = nav_cost + 1 # Nav + Communicate

                # Option 2: Have sample, need analyse
                if (rover, w) in current_have_rock:
                     nav_cost1 = self.get_distance(p_rover, w) # Nav to w
                     nav_cost2 = self.get_distance(w, self.lander_location) # Nav from w to lander
                     if nav_cost1 != infinity and nav_cost2 != infinity:
                         cost_option2 = nav_cost1 + 1 + nav_cost2 + 1 # Nav1 + Analyse + Nav2 + Communicate

                # Option 3: Sample exists, need take sample + analyse
                if w in self.at_sample_locations['rock']:
                    drop_cost = 0
                    if current_store_status.get(store) == 'full':
                        drop_cost = 1 # Drop at current location

                    nav_cost1 = self.get_distance(p_rover, w) # Nav to w
                    nav_cost2 = self.get_distance(w, self.lander_location) # Nav from w to lander

                    if nav_cost1 != infinity and nav_cost2 != infinity:
                         cost_option3 = drop_cost + nav_cost1 + 1 # Drop (if needed) + Nav to w + Take Sample
                         cost_option3 += 1 # Analyse
                         cost_option3 += nav_cost2 + 1 # Nav from w + Communicate

                current_rover_goal_cost = min(cost_option1, cost_option2, cost_option3)
                min_goal_cost = min(min_goal_cost, current_rover_goal_cost)

            # Add cost for this goal
            if min_goal_cost == infinity:
                 total_cost += IMPOSSIBLE_GOAL_PENALTY # Large penalty
            else:
                 total_cost += min_goal_cost


        # Cost for Image Goals
        for o, m in image_goals:
            if (o, m) in current_communicated_image:
                continue # Goal already met

            min_goal_cost = infinity

            # Check all imaging-equipped rovers
            for rover, capabilities in self.rover_capabilities.items():
                if 'equipped_for_imaging' not in capabilities:
                    continue

                p_rover = current_rover_locations.get(rover)
                if p_rover is None: continue

                # Find cameras on this rover supporting this mode
                suitable_cameras = []
                for cam_name, cam_props in self.camera_info.items():
                    if cam_props.get('on_board') == rover and m in cam_props.get('supports', set()):
                        suitable_cameras.append((cam_name, cam_props))

                if not suitable_cameras: continue # No suitable camera on this rover

                for cam_name, cam_props in suitable_cameras:
                    current_rover_cam_goal_cost = infinity

                    cost_option1 = infinity # Have Image -> Communicate
                    cost_option2 = infinity # Calibrated -> Take Image -> Communicate
                    cost_option3 = infinity # Need Calibrate -> Calibrate -> Take Image -> Communicate

                    # Option 1: Already have image
                    if (rover, o, m) in current_have_image:
                        nav_cost = self.get_distance(p_rover, self.lander_location)
                        if nav_cost != infinity:
                            cost_option1 = nav_cost + 1 # Nav + Communicate

                    # Option 2: Calibrated, need take image
                    if (cam_name, rover) in current_calibrated:
                        visible_wps_img = self.objective_visibility.get(o, set())
                        if visible_wps_img:
                            min_nav_img_to_lander = infinity
                            for w_img in visible_wps_img:
                                nav_cost1 = self.get_distance(p_rover, w_img) # Nav to image wp
                                nav_cost2 = self.get_distance(w_img, self.lander_location) # Nav from image wp to lander
                                if nav_cost1 != infinity and nav_cost2 != infinity:
                                    min_nav_img_to_lander = min(min_nav_img_to_lander, nav_cost1 + nav_cost2)

                            if min_nav_img_to_lander != infinity:
                                 cost_option2 = min_nav_img_to_lander + 1 + 1 # Navs + Take Image + Communicate

                    # Option 3: Need calibrate + take image
                    else:
                        obj_cal = cam_props.get('calibration_target')
                        if obj_cal:
                            visible_wps_cal = self.objective_visibility.get(obj_cal, set())
                            visible_wps_img = self.objective_visibility.get(o, set())
                            if visible_wps_cal and visible_wps_img:
                                min_nav_cal_img_to_lander = infinity
                                for w_cal in visible_wps_cal:
                                    for w_img in visible_wps_img:
                                        nav_cost1 = self.get_distance(p_rover, w_cal) # Nav to cal wp
                                        nav_cost2 = self.get_distance(w_cal, w_img) # Nav from cal wp to img wp
                                        nav_cost3 = self.get_distance(w_img, self.lander_location) # Nav from img wp to lander

                                        if nav_cost1 != infinity and nav_cost2 != infinity and nav_cost3 != infinity:
                                             min_nav_cal_img_to_lander = min(min_nav_cal_img_to_lander, nav_cost1 + nav_cost2 + nav_cost3)

                                if min_nav_cal_img_to_lander != infinity:
                                     cost_option3 = min_nav_cal_img_to_lander + 1 + 1 + 1 # Navs + Calibrate + Take Image + Communicate

                    current_rover_cam_goal_cost = min(cost_option1, cost_option2, cost_option3)
                    min_goal_cost = min(min_goal_cost, current_rover_cam_goal_cost)

            # Add cost for this goal
            if min_goal_cost == infinity:
                 total_cost += IMPOSSIBLE_GOAL_PENALTY # Large penalty
            else:
                 total_cost += min_goal_cost

        return total_cost

