from fnmatch import fnmatch
from collections import defaultdict, deque
# Assuming heuristics.heuristic_base exists and provides Heuristic class
# from heuristics.heuristic_base import Heuristic

# Define dummy Heuristic base class if not provided in the environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

def get_parts(fact):
    """
    Helper function to extract the components of a PDDL fact string.
    Removes parentheses and splits the string by spaces.
    Example: "(at rover1 waypoint1)" -> ["at", "rover1", "waypoint1"]
    """
    return fact[1:-1].split()

def match(fact, *args):
    """
    Helper function to check if a PDDL fact string matches a given pattern.
    Uses fnmatch for wildcard matching.
    Example: match("(at rover1 waypoint1)", "at", "*", "waypoint1") -> True
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the number of actions required to reach the goal state by summing
    up the estimated costs for each unachieved goal fact. The cost for each
    goal fact is estimated independently, considering the minimum cost across
    available rovers and necessary locations (sample sites, image sites,
    calibration sites, communication sites). Navigation costs are precomputed
    using BFS on the rover traversal graph.

    This heuristic is not admissible but aims to be informative for greedy
    best-first search.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing navigation distances and
        extracting static information about the domain.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Collect all unique waypoints mentioned in the problem
        waypoints = set()
        rovers = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == "at":
                rovers.add(parts[1])
                waypoints.add(parts[2])
            elif parts[0] == "at_lander":
                waypoints.add(parts[2])

        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == "can_traverse":
                rovers.add(parts[1])
                waypoints.add(parts[2])
                waypoints.add(parts[3])
            elif parts[0] == "visible":
                waypoints.add(parts[1])
                waypoints.add(parts[2])
            elif parts[0] == "visible_from":
                waypoints.add(parts[2])
            elif parts[0] == "on_board":
                 rovers.add(parts[2])
            elif parts[0] == "store_of":
                 rovers.add(parts[2])
            elif parts[0] in ["equipped_for_soil_analysis", "equipped_for_rock_analysis", "equipped_for_imaging"]:
                 rovers.add(parts[1])


        # Precompute navigation graph and shortest paths for each rover
        self.rover_graphs = defaultdict(dict) # rover -> {waypoint -> {neighbor -> cost}}
        self.rover_distances = defaultdict(dict) # rover -> {waypoint -> {waypoint -> distance}}

        # Initialize graph nodes for all waypoints for each rover
        for rover in rovers:
             self.rover_graphs[rover] = {w: {} for w in waypoints}

        # Add edges to the graph based on can_traverse and visible facts
        for fact in self.static_facts:
            if match(fact, "can_traverse", "?r", "?u", "?v"):
                r, u, v = get_parts(fact)[1:]
                # Navigation requires both can_traverse and visible
                if f"(visible {u} {v})" in self.static_facts:
                     # Ensure u and v are valid waypoints in our collected set
                     if u in waypoints and v in waypoints:
                         self.rover_graphs[r][u][v] = 1 # Cost is 1 per step

        # Compute all-pairs shortest paths using BFS for each rover
        for rover, graph in self.rover_graphs.items():
            self.rover_distances[rover] = {}
            # BFS needs to start from all nodes to get all-pairs
            for start_node in graph:
                self.rover_distances[rover][start_node] = self._bfs(graph, start_node)

        # Identify lander locations (assuming static)
        self.lander_locations = {get_parts(fact)[2] for fact in self.static_facts if match(fact, "at_lander", "*", "*")}

        # Identify communication waypoints (waypoints visible from any lander location)
        self.comm_waypoints = set()
        for lander_w in self.lander_locations:
             for fact in self.static_facts:
                 if match(fact, "visible", "?w", lander_w):
                     self.comm_waypoints.add(get_parts(fact)[1])

        # Precompute min distance from any waypoint to any communication waypoint for each rover
        self.min_dist_to_comm_waypoint = defaultdict(dict) # rover -> {waypoint -> min_distance}
        for rover in rovers:
            self.min_dist_to_comm_waypoint[rover] = {}
            for start_w in waypoints:
                min_dist = float('inf')
                # Only check distances if the start_w is a valid node in the rover's graph
                if start_w in self.rover_distances[rover]:
                    for comm_w in self.comm_waypoints:
                        if comm_w in self.rover_distances[rover][start_w]:
                             min_dist = min(min_dist, self.rover_distances[rover][start_w][comm_w])
                self.min_dist_to_comm_waypoint[rover][start_w] = min_dist


        # Identify objective visibility waypoints
        self.objective_waypoints = defaultdict(set) # objective -> {waypoint}
        for fact in self.static_facts:
            if match(fact, "visible_from", "?o", "?w"):
                o, w = get_parts(fact)[1:]
                self.objective_waypoints[o].add(w)

        # Identify calibration targets and visibility waypoints
        self.calibration_targets = {} # camera -> objective
        self.calibration_waypoints = defaultdict(set) # camera -> {waypoint}
        for fact in self.static_facts:
            if match(fact, "calibration_target", "?i", "?t"):
                i, t = get_parts(fact)[1:]
                self.calibration_targets[i] = t
                # Find waypoints visible from the calibration target objective
                if t in self.objective_waypoints:
                     self.calibration_waypoints[i].update(self.objective_waypoints[t])

        # Identify rover capabilities, cameras, modes, and stores
        self.rover_capabilities = defaultdict(dict) # rover -> {capability_type -> bool}
        self.rover_cameras = defaultdict(set) # rover -> {camera}
        self.camera_modes = defaultdict(set) # camera -> {mode}
        self.rover_stores = defaultdict(set) # rover -> {store}

        for fact in self.static_facts:
            if match(fact, "equipped_for_soil_analysis", "?r"):
                self.rover_capabilities[get_parts(fact)[1]]["soil"] = True
            elif match(fact, "equipped_for_rock_analysis", "?r"):
                self.rover_capabilities[get_parts(fact)[1]]["rock"] = True
            elif match(fact, "equipped_for_imaging", "?r"):
                self.rover_capabilities[get_parts(fact)[1]]["imaging"] = True
            elif match(fact, "on_board", "?i", "?r"):
                self.rover_cameras[get_parts(fact)[2]].add(get_parts(fact)[1])
            elif match(fact, "supports", "?i", "?m"):
                self.camera_modes[get_parts(fact)[1]].add(get_parts(fact)[2])
            elif match(fact, "store_of", "?s", "?r"):
                 self.rover_stores[get_parts(fact)[2]].add(get_parts(fact)[1])


    def _bfs(self, graph, start_node):
        """
        Performs Breadth-First Search to find shortest distances from a start_node
        in a given graph.
        """
        distances = {node: float('inf') for node in graph}
        if start_node in graph: # Ensure start_node is a valid key
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()

                if current_node in graph: # Ensure current_node is a key in the graph
                    for neighbor in graph[current_node]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = distances[current_node] + 1
                            queue.append(neighbor)
        return distances

    def get_distance(self, rover, start_w, end_w):
        """
        Gets the shortest distance (number of navigation steps) for a rover
        between two waypoints. Returns float('inf') if unreachable.
        """
        if rover in self.rover_distances and start_w in self.rover_distances[rover] and end_w in self.rover_distances[rover][start_w]:
            return self.rover_distances[rover][start_w][end_w]
        return float('inf') # Cannot traverse

    def get_rover_location(self, state, rover):
        """Find the current waypoint of a rover in the state."""
        for fact in state:
            if match(fact, "at", rover, "?w"):
                return get_parts(fact)[2]
        return None # Should not happen in valid states

    def get_rover_store(self, state, rover):
         """Find the store of a rover (assuming one store per rover)."""
         # Store info is static, but empty/full status is not
         for store in self.rover_stores.get(rover, []):
             return store # Assuming one store per rover
         return None # Should not happen if rover has store_of fact

    def is_store_full(self, state, store):
        """Check if a store is full in the state."""
        return f"(full {store})" in state

    def is_camera_calibrated(self, state, camera, rover):
        """Check if a camera on a rover is calibrated in the state."""
        return f"(calibrated {camera} {rover})" in state

    def has_soil_sample(self, state, rover, waypoint):
        """Check if a rover has a soil sample from a waypoint."""
        return f"(have_soil_analysis {rover} {waypoint})" in state

    def has_rock_sample(self, state, rover, waypoint):
        """Check if a rover has a rock sample from a waypoint."""
        return f"(have_rock_analysis {rover} {waypoint})" in state

    def has_image(self, state, rover, objective, mode):
        """Check if a rover has an image of an objective in a mode."""
        return f"(have_image {rover} {objective} {mode})" in state


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        Estimates the sum of minimum costs for each unachieved goal fact.
        """
        state = node.state
        final_total_cost = 0

        # Group goals by type for easier processing
        goals_by_type = defaultdict(list)
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "communicated_soil_data":
                goals_by_type["soil"].append(parts[1]) # waypoint
            elif parts[0] == "communicated_rock_data":
                goals_by_type["rock"].append(parts[1]) # waypoint
            elif parts[0] == "communicated_image_data":
                goals_by_type["image"].append((parts[1], parts[2])) # (objective, mode)

        # Process soil data goals
        for waypoint in goals_by_type["soil"]:
            # If goal is already achieved, cost is 0
            if f"(communicated_soil_data {waypoint})" in state:
                continue

            cost_for_this_goal = float('inf')

            # Option 1: Communicate existing sample
            rover_with_sample = None
            for fact in state:
                if match(fact, "have_soil_analysis", "?r", waypoint):
                    rover_with_sample = get_parts(fact)[1]
                    break

            if rover_with_sample:
                # Sample is already collected, just need to communicate
                r = rover_with_sample
                r_location = self.get_rover_location(state, r)
                if r_location:
                    # Cost is navigation from current location to nearest comm point + communicate action
                    nav_to_comm = self.min_dist_to_comm_waypoint[r].get(r_location, float('inf'))
                    if nav_to_comm != float('inf'):
                        cost_for_this_goal = min(cost_for_this_goal, nav_to_comm + 1) # +1 for communicate action

            # Option 2: Sample and then communicate
            # Check if sample is still available at the waypoint
            if f"(at_soil_sample {waypoint})" in state:
                # Find a suitable rover equipped for soil analysis
                for fact in self.static_facts:
                    if match(fact, "equipped_for_soil_analysis", "?r"):
                        r = get_parts(fact)[1]
                        r_location = self.get_rover_location(state, r)
                        store = self.get_rover_store(state, r)

                        if r_location and store:
                            cost_to_sample = 0
                            # Cost to empty store if full
                            if self.is_store_full(state, store):
                                cost_to_sample += 1 # drop action

                            # Cost to navigate to sample waypoint
                            nav_to_sample_cost = self.get_distance(r, r_location, waypoint)
                            if nav_to_sample_cost == float('inf'):
                                continue # Rover cannot reach sample waypoint

                            cost_to_sample += nav_to_sample_cost + 1 # +1 for sample_soil action

                            # Cost to communicate from sample waypoint
                            nav_to_comm = self.min_dist_to_comm_waypoint[r].get(waypoint, float('inf')) # Navigate from sample waypoint
                            if nav_to_comm != float('inf'):
                                cost_to_communicate = nav_to_comm + 1 # +1 for communicate action
                                cost_for_this_goal = min(cost_for_this_goal, cost_to_sample + cost_to_communicate)

            # Add the minimum cost for this goal to the total
            if cost_for_this_goal != float('inf'):
                final_total_cost += cost_for_this_goal


        # Process rock data goals (analogous to soil)
        for waypoint in goals_by_type["rock"]:
            # If goal is already achieved, cost is 0
            if f"(communicated_rock_data {waypoint})" in state:
                continue

            cost_for_this_goal = float('inf')

            # Option 1: Communicate existing sample
            rover_with_sample = None
            for fact in state:
                if match(fact, "have_rock_analysis", "?r", waypoint):
                    rover_with_sample = get_parts(fact)[1]
                    break

            if rover_with_sample:
                # Sample is already collected, just need to communicate
                r = rover_with_sample
                r_location = self.get_rover_location(state, r)
                if r_location:
                    # Cost is navigation from current location to nearest comm point + communicate action
                    nav_to_comm = self.min_dist_to_comm_waypoint[r].get(r_location, float('inf'))
                    if nav_to_comm != float('inf'):
                        cost_for_this_goal = min(cost_for_this_goal, nav_to_comm + 1) # +1 for communicate action

            # Option 2: Sample and then communicate
            # Check if sample is still available at the waypoint
            if f"(at_rock_sample {waypoint})" in state:
                # Find a suitable rover equipped for rock analysis
                for fact in self.static_facts:
                    if match(fact, "equipped_for_rock_analysis", "?r"):
                        r = get_parts(fact)[1]
                        r_location = self.get_rover_location(state, r)
                        store = self.get_rover_store(state, r)

                        if r_location and store:
                            cost_to_sample = 0
                            # Cost to empty store if full
                            if self.is_store_full(state, store):
                                cost_to_sample += 1 # drop action

                            # Cost to navigate to sample waypoint
                            nav_to_sample_cost = self.get_distance(r, r_location, waypoint)
                            if nav_to_sample_cost == float('inf'):
                                continue # Rover cannot reach sample waypoint

                            cost_to_sample += nav_to_sample_cost + 1 # +1 for sample_rock action

                            # Cost to communicate from sample waypoint
                            nav_to_comm = self.min_dist_to_comm_waypoint[r].get(waypoint, float('inf')) # Navigate from sample waypoint
                            if nav_to_comm != float('inf'):
                                cost_to_communicate = nav_to_comm + 1 # +1 for communicate action
                                cost_for_this_goal = min(cost_for_this_goal, cost_to_sample + cost_to_communicate)

            # Add the minimum cost for this goal to the total
            if cost_for_this_goal != float('inf'):
                final_total_cost += cost_for_this_goal


        # Process image data goals
        for objective, mode in goals_by_type["image"]:
            # If goal is already achieved, cost is 0
            if f"(communicated_image_data {objective} {mode})" in state:
                continue

            cost_for_this_goal = float('inf')

            # Option 1: Communicate existing image
            rover_with_image = None
            for fact in state:
                if match(fact, "have_image", "?r", objective, mode):
                    rover_with_image = get_parts(fact)[1]
                    break

            if rover_with_image:
                # Image is already taken, just need to communicate
                r = rover_with_image
                r_location = self.get_rover_location(state, r)
                if r_location:
                    # Cost is navigation from current location to nearest comm point + communicate action
                    nav_to_comm = self.min_dist_to_comm_waypoint[r].get(r_location, float('inf'))
                    if nav_to_comm != float('inf'):
                        cost_for_this_goal = min(cost_for_this_goal, nav_to_comm + 1) # +1 for communicate action

            # Option 2: Take image and then communicate
            # Find a suitable rover with imaging equipment and camera supporting the mode
            for fact in self.static_facts:
                 if match(fact, "equipped_for_imaging", "?r"):
                     r = get_parts(fact)[1]
                     r_location = self.get_rover_location(state, r)
                     if not r_location: continue # Rover location unknown? Skip.

                     # Find cameras on this rover supporting the mode
                     suitable_cameras = [
                         cam for cam in self.rover_cameras.get(r, [])
                         if mode in self.camera_modes.get(cam, [])
                     ]

                     for i in suitable_cameras:
                         image_waypoints = self.objective_waypoints.get(objective, set())
                         if not image_waypoints: continue # Cannot image this objective

                         cal_target = self.calibration_targets.get(i)
                         cal_waypoints = self.calibration_waypoints.get(i, set())
                         needs_calibration = not self.is_camera_calibrated(state, i, r)

                         # Estimate cost to take image and communicate for this rover/camera
                         cost_to_take_image_and_comm_rc = float('inf')

                         if needs_calibration:
                             if not cal_waypoints: continue # Cannot calibrate

                             # Need to visit a cal_w and an img_w, then a comm_w
                             # Path: current -> cal_w -> img_w -> comm_w
                             # Minimize dist(r_loc, w) + dist(w, p) + dist(p, comm) over w, p, comm
                             # Optimized: min_{w in CalWs} (dist(r_loc, w) + min_{p in ImgWs} (dist(w, p) + min_dist_to_comm(p, r)))
                             min_nav_cal_img_comm = float('inf')

                             for cal_w in cal_waypoints:
                                 nav_r_to_cal = self.get_distance(r, r_location, cal_w)
                                 if nav_r_to_cal == float('inf'): continue

                                 min_nav_cal_to_img_then_comm = float('inf')
                                 for img_w in image_waypoints:
                                     nav_cal_to_img = self.get_distance(r, cal_w, img_w)
                                     if nav_cal_to_img == float('inf'): continue

                                     nav_img_to_comm = self.min_dist_to_comm_waypoint[r].get(img_w, float('inf'))
                                     if nav_img_to_comm == float('inf'): continue

                                     # Cost from cal_w: nav_cal_to_img + nav_img_to_comm
                                     min_nav_cal_to_img_then_comm = min(min_nav_cal_to_img_then_comm, nav_cal_to_img + nav_img_to_comm)

                                 if min_nav_cal_to_img_then_comm != float('inf'):
                                     # Total nav cost: nav_r_to_cal + min_nav_cal_to_img_then_comm
                                     min_nav_cal_img_comm = min(min_nav_cal_img_comm, nav_r_to_cal + min_nav_cal_to_img_then_comm)


                             if min_nav_cal_img_comm != float('inf'):
                                 # Total cost: nav + calibrate + take_image + communicate
                                 cost_to_take_image_and_comm_rc = min_nav_cal_img_comm + 1 + 1 + 1


                         else: # Already calibrated
                             # Need to visit an img_w, then a comm_w
                             # Path: current -> img_w -> comm_w
                             # Minimize dist(r_loc, p) + dist(p, comm) over p, comm
                             # Optimized: min_{p in ImgWs} (dist(r_loc, p) + min_dist_to_comm(p, r))
                             min_nav_img_comm = float('inf')

                             for img_w in image_waypoints:
                                 nav_r_to_img = self.get_distance(r, r_location, img_w)
                                 if nav_r_to_img == float('inf'): continue

                                 nav_img_to_comm = self.min_dist_to_comm_waypoint[r].get(img_w, float('inf'))
                                 if nav_img_to_comm == float('inf'): continue

                                 # Total nav cost: nav_r_to_img + nav_img_to_comm
                                 min_nav_img_comm = min(min_nav_img_comm, nav_r_to_img + nav_img_to_comm)

                             if min_nav_img_comm != float('inf'):
                                 # Total cost: nav + take_image + communicate
                                 cost_to_take_image_and_comm_rc = min_nav_img_comm + 1 + 1

                         # Update the minimum cost for this goal across all suitable rovers/cameras
                         cost_for_this_goal = min(cost_for_this_goal, cost_to_take_image_and_comm_rc)


            # Add the minimum cost for this goal to the total
            if cost_for_this_goal != float('inf'):
                final_total_cost += cost_for_this_goal


        # Return the total estimated cost
        return final_total_cost

