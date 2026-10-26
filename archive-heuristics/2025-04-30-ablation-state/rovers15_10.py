from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this exists

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings from split if there are multiple spaces
    return [part for part in fact[1:-1].split() if part]

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

# BFS for shortest path on the navigation graph
def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest paths from a start node
    to all other nodes in a graph.

    Args:
        graph: Adjacency list representation {node: [neighbor1, neighbor2, ...]}
        start_node: The node to start the BFS from.

    Returns:
        A dictionary {node: distance} containing shortest distances.
        Distance is float('inf') if a node is unreachable.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node might not be in the graph if it's an object type not a waypoint
         # or if the graph was built only from traversable waypoints and the start
         # is an isolated waypoint.
         return distances # All unreachable

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current = queue.popleft()
        # Ensure current is a valid key in graph
        if current in graph:
            for neighbor in graph[current]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
    return distances

def min_dist_sets(dist_matrix, set1, set2):
    """
    Finds the minimum shortest path distance between any node in set1 and any node in set2.

    Args:
        dist_matrix: The precomputed APSP distance matrix {node1: {node2: distance}}.
        set1: A set of start nodes.
        set2: A set of end nodes.

    Returns:
        The minimum distance, or float('inf') if no node in set2 is reachable from any node in set1.
    """
    min_d = float('inf')
    for w1 in set1:
        if w1 in dist_matrix:
            for w2 in set2:
                min_d = min(min_d, dist_matrix[w1].get(w2, float('inf')))
    return min_d


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    unmet goal conditions. It sums the estimated costs for each individual
    unmet goal. The cost for a goal is estimated based on the minimum actions
    and navigation steps required for a suitable rover to:
    1. Obtain the necessary data (sample or image).
    2. Move to a location visible from the lander.
    3. Communicate the data.

    Navigation costs are estimated using precomputed shortest path distances
    on the waypoint graph. Action costs (sample, drop, calibrate, take_image,
    communicate) are counted as 1. The heuristic is non-admissible as it
    ignores negative interactions (like store capacity, camera calibration state
    after taking an image, or sample depletion) and sums costs for goals
    independently.

    # Assumptions
    - The navigation graph (based on `visible` and `can_traverse` facts) is static.
    - Shortest path distances between waypoints can be precomputed.
    - Any rover equipped for a task can potentially perform it.
    - Any camera on a rover supporting a mode can be used for that mode.
    - Calibration is possible if a calibration target and a visible waypoint for it exist.
    - Sampling is possible if the sample exists at the waypoint.
    - Communication is possible if the rover is at a waypoint visible from the lander.
    - Store capacity is considered simply (1 action cost if store is full before sampling).
    - Camera calibration state is considered simply (1 action cost if camera needs calibration before taking image).
    - The heuristic sums costs for obtaining data and communicating independently, potentially double-counting navigation or ignoring required sequences (e.g., calibrate -> image -> communicate).

    # Heuristic Initialization
    The constructor precomputes and stores static information:
    - The lander's waypoint.
    - Sets of rovers equipped for soil, rock, and imaging tasks.
    - Mapping from rovers to their stores.
    - Information about cameras (which rover they are on, which modes they support, their calibration target).
    - Mapping from objectives/calibration targets to waypoints from which they are visible.
    - The navigation graph based on `visible` and `can_traverse` facts.
    - All-pairs shortest path distances between waypoints using BFS.
    - The set of waypoints visible from the lander (communication waypoints).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of each rover.
    2. Identify which stores are full.
    3. Identify which data/images the rovers currently possess.
    4. Identify which cameras are currently calibrated.
    5. Identify which soil/rock samples still exist at waypoints.
    6. Initialize total heuristic cost to 0.
    7. Iterate through each goal fact:
        a. If the goal fact is already true in the current state, add 0 cost for this goal.
        b. If the goal is `(communicated_soil_data ?w)`:
            - Find the minimum cost among all soil-equipped rovers `r`.
            - Cost for rover `r`:
                - If `r` already has `(have_soil_analysis r w)`:
                    - Cost = `min_dist(at(r), W_comm)` + 1 (communicate).
                - If `r` does not have the sample:
                    - If `(at_soil_sample w)` is false, this goal is impossible via sampling; cost is infinity.
                    - Cost to get sample = `dist(at(r), w)` + 1 (sample) + (1 if `r`'s store is full).
                    - Cost to communicate = `min_dist(w, W_comm)` + 1 (communicate). (Assumes communication from sample location).
                    - Total cost = Cost to get sample + Cost to communicate.
            - Add the minimum cost over all suitable rovers to the total heuristic. If no suitable rover or path, add infinity.
        c. If the goal is `(communicated_rock_data ?w)`:
            - Similar logic as soil data, using rock-equipped rovers and rock samples.
        d. If the goal is `(communicated_image_data ?o ?m)`:
            - Find the minimum cost among all imaging-equipped rovers `r`.
            - Cost for rover `r`:
                - If `r` already has `(have_image r o m)`:
                    - Cost = `min_dist(at(r), W_comm)` + 1 (communicate).
                - If `r` does not have the image:
                    - Find a camera `i` on `r` supporting `m`. If none, cost is infinity for this rover.
                    - Find cal target `t` for `i` and cal waypoints `W_cal_t`. If none, cost is infinity.
                    - Find image waypoints `W_img_o`. If none, cost is infinity.
                    - If calibration is needed (`(i, r)` not calibrated):
                        - Cost to get image = `min_dist(at(r), W_cal_t)` + 1 (calibrate) + `min_dist(W_cal_t, W_img_o)` + 1 (take image). (Nav from current to cal, cal, nav from cal to img, img).
                    - If calibration is not needed (`(i, r)` calibrated):
                        - Cost to get image = `min_dist(at(r), W_img_o)` + 1 (take image). (Nav from current to img, img).
                    - Cost to communicate = `min_dist(W_img_o, W_comm)` + 1 (communicate). (Nav from img to comm, comm).
                    - Total cost = Cost to get image + Cost to communicate.
            - Add the minimum cost over all suitable rovers to the total heuristic. If no suitable rover/camera/waypoint or path, add infinity.
    8. Return the total heuristic cost. If it's infinity, return a large number.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        infinity = float('inf')

        # 1. Parse static facts
        self.lander_waypoint = None
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.rover_stores = {} # {rover: store}
        self.camera_info = {} # {camera: {'rover': r, 'modes': {m}, 'cal_target': t}}
        self.objective_waypoints = {} # {objective/cal_target: {waypoint}}
        self.waypoints = set()
        can_traverse_edges = set() # {(w_from, w_to)}
        visible_facts = set() # {(w1, w2)}

        # Collect all potential waypoints and static info
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            pred = parts[0]
            if pred == 'at_lander':
                self.lander_waypoint = parts[2] if len(parts) > 2 else None
                if self.lander_waypoint: self.waypoints.add(self.lander_waypoint)
            elif pred == 'at': # Rovers' initial locations are static
                 if len(parts) > 2 and parts[1].startswith('rover'):
                     self.waypoints.add(parts[2])
            elif pred == 'at_soil_sample' or pred == 'at_rock_sample':
                 if len(parts) > 1: self.waypoints.add(parts[1])
            elif pred == 'can_traverse':
                if len(parts) > 3:
                    self.waypoints.add(parts[2])
                    self.waypoints.add(parts[3])
                    can_traverse_edges.add((parts[2], parts[3]))
            elif pred == 'visible':
                if len(parts) > 2:
                    self.waypoints.add(parts[1])
                    self.waypoints.add(parts[2])
                    visible_facts.add((parts[1], parts[2]))
            elif pred == 'equipped_for_soil_analysis':
                if len(parts) > 1: self.equipped_soil.add(parts[1])
            elif pred == 'equipped_for_rock_analysis':
                if len(parts) > 1: self.equipped_rock.add(parts[1])
            elif pred == 'equipped_for_imaging':
                if len(parts) > 1: self.equipped_imaging.add(parts[1])
            elif pred == 'store_of':
                if len(parts) > 2: self.rover_stores[parts[2]] = parts[1]
            elif pred == 'on_board':
                if len(parts) > 2:
                    camera, rover = parts[1], parts[2]
                    if camera not in self.camera_info: self.camera_info[camera] = {}
                    self.camera_info[camera]['rover'] = rover
            elif pred == 'supports':
                if len(parts) > 2:
                    camera, mode = parts[1], parts[2]
                    if camera not in self.camera_info: self.camera_info[camera] = {}
                    if 'modes' not in self.camera_info[camera]: self.camera_info[camera]['modes'] = set()
                    self.camera_info[camera]['modes'].add(mode)
            elif pred == 'calibration_target':
                if len(parts) > 2:
                    camera, target = parts[1], parts[2]
                    if camera not in self.camera_info: self.camera_info[camera] = {}
                    self.camera_info[camera]['cal_target'] = target
            elif pred == 'visible_from':
                if len(parts) > 2:
                    obj, waypoint = parts[1], parts[2]
                    if obj not in self.objective_waypoints: self.objective_waypoints[obj] = set()
                    self.objective_waypoints[obj].add(waypoint)
                    self.waypoints.add(waypoint)

        # Ensure all waypoints from goals are included (e.g., sample locations)
        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue
             pred = parts[0]
             if pred in ['communicated_soil_data', 'communicated_rock_data']:
                 if len(parts) > 1: self.waypoints.add(parts[1])
             # communicated_image_data refers to objective/mode, waypoints come from visible_from

        # 2. Build navigation graph
        self.adj = {w: set() for w in self.waypoints}
        for w1, w2 in visible_facts:
             # An edge exists if visible AND traversable by *some* rover
             if (w1, w2) in can_traverse_edges:
                 if w1 in self.adj: self.adj[w1].add(w2)
                 if w2 in self.adj: self.adj[w2].add(w1) # Assuming symmetric traversal for graph building

        # 3. Compute APSP
        self.dist = {w: bfs(self.adj, w) for w in self.waypoints}

        # 4. Identify communication waypoints
        self.comm_waypoints = {w for w in self.waypoints if self.lander_waypoint and (w, self.lander_waypoint) in visible_facts}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0
        infinity = float('inf') # Use float('inf') internally
        large_finite = 1e9 # Value to return for unreachable goals

        # 1. Extract current state information
        rover_locations = {} # {rover: waypoint}
        full_stores = set() # {store}
        have_soil = set() # {(rover, waypoint)}
        have_rock = set() # {(rover, waypoint)}
        have_image = set() # {(rover, objective, mode)}
        calibrated_cameras = set() # {(camera, rover)}
        soil_samples_at = set() # {waypoint}
        rock_samples_at = set() # {waypoint}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at' and parts[1].startswith('rover'):
                if len(parts) > 2: rover_locations[parts[1]] = parts[2]
            elif pred == 'full':
                if len(parts) > 1: full_stores.add(parts[1])
            elif pred == 'have_soil_analysis':
                if len(parts) > 2: have_soil.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis':
                if len(parts) > 2: have_rock.add((parts[1], parts[2]))
            elif pred == 'have_image':
                if len(parts) > 3: have_image.add((parts[1], parts[2], parts[3]))
            elif pred == 'calibrated':
                if len(parts) > 2: calibrated_cameras.add((parts[1], parts[2]))
            elif pred == 'at_soil_sample':
                 if len(parts) > 1: soil_samples_at.add(parts[1])
            elif pred == 'at_rock_sample':
                 if len(parts) > 1: rock_samples_at.add(parts[1])

        # 2. Iterate through goals and sum costs
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]

            min_goal_cost = infinity

            if pred == "communicated_soil_data":
                if len(parts) < 2: continue # Malformed goal
                w = parts[1]
                for r in self.equipped_soil:
                    rover_loc = rover_locations.get(r)
                    if rover_loc is None: continue # Rover not in state? Should not happen.
                    store = self.rover_stores.get(r)
                    if store is None: continue # Rover has no store? Should not happen.

                    cost_get_sample = 0
                    if (r, w) not in have_soil:
                        if w not in soil_samples_at:
                            # Sample is gone, this rover cannot get it via sampling
                            cost_get_sample = infinity
                        else:
                            dist_to_sample = self.dist.get(rover_loc, {}).get(w, infinity)
                            if dist_to_sample == infinity:
                                cost_get_sample = infinity # Cannot reach sample location
                            else:
                                cost_get_sample = dist_to_sample + 1 # navigate + sample
                                if store in full_stores:
                                    cost_get_sample += 1 # drop

                    if cost_get_sample == infinity:
                         continue # Cannot get sample, so cannot communicate it

                    cost_communicate = 0
                    # Rover needs to be at a comm waypoint. If it just sampled, it's at w.
                    # If it already had the sample, it's at rover_loc.
                    current_loc_for_comm = w if (r, w) not in have_soil else rover_loc

                    min_dist_to_comm = infinity
                    if current_loc_for_comm in self.dist and self.comm_waypoints:
                        min_dist_to_comm = min_dist_sets(self.dist, {current_loc_for_comm}, self.comm_waypoints)

                    if min_dist_to_comm == infinity:
                        cost_communicate = infinity # Cannot reach any comm waypoint
                    else:
                        cost_communicate = min_dist_to_comm + 1 # navigate + communicate

                    if cost_communicate != infinity:
                        min_goal_cost = min(min_goal_cost, cost_get_sample + cost_communicate)


            elif pred == "communicated_rock_data":
                if len(parts) < 2: continue # Malformed goal
                w = parts[1]
                for r in self.equipped_rock:
                    rover_loc = rover_locations.get(r)
                    if rover_loc is None: continue
                    store = self.rover_stores.get(r)
                    if store is None: continue

                    cost_get_sample = 0
                    if (r, w) not in have_rock:
                        if w not in rock_samples_at:
                            cost_get_sample = infinity
                        else:
                            dist_to_sample = self.dist.get(rover_loc, {}).get(w, infinity)
                            if dist_to_sample == infinity:
                                cost_get_sample = infinity
                            else:
                                cost_get_sample = dist_to_sample + 1 # navigate + sample
                                if store in full_stores:
                                    cost_get_sample += 1 # drop

                    if cost_get_sample == infinity:
                         continue

                    cost_communicate = 0
                    current_loc_for_comm = w if (r, w) not in have_rock else rover_loc

                    min_dist_to_comm = infinity
                    if current_loc_for_comm in self.dist and self.comm_waypoints:
                         min_dist_to_comm = min_dist_sets(self.dist, {current_loc_for_comm}, self.comm_waypoints)

                    if min_dist_to_comm == infinity:
                        cost_communicate = infinity
                    else:
                        cost_communicate = min_dist_to_comm + 1 # navigate + communicate

                    if cost_communicate != infinity:
                        min_goal_cost = min(min_goal_cost, cost_get_sample + cost_communicate)


            elif pred == "communicated_image_data":
                if len(parts) < 3: continue # Malformed goal
                o, m = parts[1], parts[2]
                for r in self.equipped_imaging:
                    rover_loc = rover_locations.get(r)
                    if rover_loc is None: continue

                    cost_get_image = 0
                    if (r, o, m) not in have_image:
                        # Need to take image
                        cost_get_image += 1 # take_image action

                        # Find suitable camera on r supporting m
                        suitable_camera = None
                        for cam, info in self.camera_info.items():
                            if info.get('rover') == r and m in info.get('modes', set()):
                                suitable_camera = cam
                                break # Assume the first found is sufficient for heuristic

                        if suitable_camera is None:
                             cost_get_image = infinity # No camera supports mode m on this rover
                        else:
                            # Check if calibration is possible for this camera
                            cal_target = self.camera_info[suitable_camera].get('cal_target')
                            cal_waypoints = self.objective_waypoints.get(cal_target, set()) if cal_target else set()

                            image_waypoints = self.objective_waypoints.get(o, set())

                            if not image_waypoints:
                                cost_get_image = infinity # No waypoint to view objective from
                            elif (suitable_camera, r) not in calibrated_cameras:
                                # Calibration needed
                                if not cal_waypoints:
                                    cost_get_image = infinity # Calibration required but impossible
                                else:
                                    cost_get_image += 1 # calibrate action
                                    # Nav from current to cal
                                    min_dist_to_cal = min_dist_sets(self.dist, {rover_loc}, cal_waypoints)
                                    # Nav from cal to img
                                    min_dist_cal_to_img = min_dist_sets(self.dist, cal_waypoints, image_waypoints)

                                    if min_dist_to_cal == infinity or min_dist_cal_to_img == infinity:
                                        cost_get_image = infinity # Cannot complete navigation for image
                                    else:
                                        cost_get_image += min_dist_to_cal + min_dist_cal_to_img
                            else:
                                # Camera is already calibrated, no calibrate action needed
                                # Nav directly from current loc to image waypoint
                                min_dist_to_img = min_dist_sets(self.dist, {rover_loc}, image_waypoints)
                                if min_dist_to_img == infinity:
                                    cost_get_image = infinity # Cannot reach image location
                                else:
                                    cost_get_image += min_dist_to_img


                    if cost_get_image == infinity:
                         continue # Cannot get image, so cannot communicate it

                    cost_communicate = 0
                    # Need to navigate to a comm waypoint from the image waypoint
                    image_waypoints = self.objective_waypoints.get(o, set()) # Need this again
                    if not image_waypoints or not self.comm_waypoints:
                         cost_communicate = infinity # Should have been caught by cost_get_image, but double check

                    min_dist_img_to_comm = infinity
                    # Only calculate if image was potentially obtainable and waypoints exist
                    if cost_get_image != infinity and image_waypoints and self.comm_waypoints:
                         min_dist_img_to_comm = min_dist_sets(self.dist, image_waypoints, self.comm_waypoints)

                    if min_dist_img_to_comm == infinity:
                        cost_communicate = infinity # Cannot reach any comm waypoint from image location
                    else:
                        cost_communicate = min_dist_img_to_comm + 1 # navigate + communicate

                    if cost_communicate != infinity:
                        min_goal_cost = min(min_goal_cost, cost_get_image + cost_communicate)


            # Add the minimum cost for this goal (over all rovers/methods) to the total
            if min_goal_cost == infinity:
                 # If any goal is impossible, the total heuristic should reflect that
                 # Returning a large number is standard for unreachable goals
                 return large_finite # A large finite number indicating unsolvability from this state

            total_cost += min_goal_cost

        # Return the total estimated cost.
        # If total_cost is 0, it means all goals were already in the state.
        return total_cost
