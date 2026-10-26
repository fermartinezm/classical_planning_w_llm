import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we have at least as many parts as non-wildcard args
    if len(parts) < len([arg for arg in args if arg != '*']):
        return False
    # Check if parts match args, allowing for trailing wildcards in args
    for i in range(min(len(parts), len(args))):
        if not fnmatch(parts[i], args[i]):
            return False
    # If args has more parts than facts, the remaining args must be wildcards
    if len(args) > len(parts):
        return all(arg == '*' for arg in args[len(parts):])
    return True


def bfs(graph, start_node):
    """
    Perform BFS on a graph to find shortest distances from a start node.

    Args:
        graph: A dictionary where keys are nodes and values are sets of neighbor nodes.
               Assumes all relevant nodes are keys, even if they have no neighbors.
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its distance from the start_node.
        Returns distances for all nodes in the graph.
    """
    distances = {node: float('inf') for node in graph}
    if start_node in graph: # Only proceed if start_node is a valid key in the graph
        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph
            if current_node in graph:
                for neighbor in graph[current_node]:
                    # Ensure neighbor is also a valid node in the graph distances dict
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    communication goals. It sums the estimated minimum cost for each unachieved
    communication goal independently. The cost for a goal includes the actions
    needed to obtain the required data (sample soil/rock, take image) and the
    action to communicate it, plus estimated navigation costs using precomputed
    shortest path distances.

    # Assumptions
    - Each unachieved communication goal is treated independently.
    - Obtaining a soil/rock sample requires 1 sample action and navigation to the sample location.
    - Obtaining an image requires 1 calibrate action, 1 take_image action, and navigation to a calibration waypoint and then to an imaging waypoint.
    - Communication requires 1 communicate action and navigation to a lander-visible waypoint.
    - The heuristic estimates navigation cost using precomputed shortest path distances (BFS) on the rover's traversal graph.
    - The heuristic assumes that if a `have_` fact is needed but not present, it will be obtained by *some* suitable rover, and that same rover will perform the communication. The minimum cost over all suitable rovers is used.
    - The heuristic ignores the `empty` store constraint for sampling and the `calibrated` state for imaging beyond the initial `calibrate` action per image goal.
    - Unreachable goals or required locations result in an infinite heuristic value for that goal component.

    # Heuristic Initialization
    The initialization phase precomputes static information from the task:
    - The lander's location.
    - Sets of rovers equipped for soil, rock, and imaging.
    - Mapping of stores to rovers.
    - Camera capabilities (on_board, supports, calibration_target).
    - Objective visibility from waypoints.
    - The traversal graph for each rover based on `can_traverse` facts.
    - Shortest path distances between all pairs of waypoints for each rover using BFS.
    - The set of waypoints visible from the lander.
    - Shortest path distances from every waypoint to any lander-visible waypoint for each rover.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. Identify all communication goals that are not yet satisfied in the current state.
    3. For each unachieved communication goal (soil, rock, or image):
        a. Calculate the minimum estimated cost to achieve *only* this goal, considering all capable rovers.
        b. For a soil/rock goal `(communicated_soil/rock_data ?w)`:
            i. Iterate through all rovers equipped for the task (soil/rock).
            ii. If the rover's current location `r_loc` is known:
                - Calculate cost to get the sample: 1 (sample action) + navigation cost from `r_loc` to `?w`. This cost is added only if the `(have_soil/rock_analysis r w)` fact is not already in the state.
                - Calculate cost to communicate: 1 (communicate action) + navigation cost from the location where the sample is obtained (either `?w` if sampled, or `r_loc` if already had) to the nearest lander-visible waypoint.
                - Sum these costs for the current rover.
                - Update the minimum cost for this goal across all considered rovers.
        c. For an image goal `(communicated_image_data ?o ?m)`:
            i. Iterate through all rovers equipped for imaging.
            ii. If the rover's current location `r_loc` is known:
                - Calculate cost to get the image: 1 (calibrate) + 1 (take_image) + navigation cost from `r_loc` to a suitable calibration waypoint + navigation cost from that calibration waypoint to a suitable imaging waypoint. This cost is added only if the `(have_image r o m)` fact is not already in the state. Finding suitable waypoints involves checking static facts (`calibration_target`, `visible_from`). Minimum navigation cost over all suitable waypoint pairs is used.
                - Calculate cost to communicate: 1 (communicate action) + navigation cost from the location where the image is obtained (an imaging waypoint if taken, or `r_loc` if already had) to the nearest lander-visible waypoint.
                - Sum these costs for the current rover.
                - Update the minimum cost for this goal across all considered rovers.
        d. Add the minimum estimated cost for this goal to the total heuristic cost.
    4. Return the total heuristic cost. Handle cases where a goal is unreachable (e.g., return float('inf')).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        all_facts = task.facts # Includes all possible facts, useful for getting all objects

        # Extract all object names by type (using all_facts for comprehensive list)
        self.rovers = {get_parts(f)[1] for f in all_facts if match(f, "rover", "*")}
        self.waypoint_facts = {f for f in all_facts if match(f, "waypoint", "*")} # Keep original fact strings
        self.waypoints = {get_parts(f)[1] for f in self.waypoint_facts}
        self.stores = {get_parts(f)[1] for f in all_facts if match(f, "store", "*")}
        self.cameras = {get_parts(f)[1] for f in all_facts if match(f, "camera", "*")}
        self.modes = {get_parts(f)[1] for f in all_facts if match(f, "mode", "*")}
        self.landers = {get_parts(f)[1] for f in all_facts if match(f, "lander", "*")}
        self.objectives = {get_parts(f)[1] for f in all_facts if match(f, "objective", "*")}


        # Extract static information
        self.lander_waypoint = None
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.store_of = {} # store -> rover
        self.on_board = {} # rover -> set(camera)
        self.supports = {} # camera -> set(mode)
        self.calibration_target = {} # camera -> set(objective)
        self.objective_visible_from = {} # objective -> set(waypoint)
        self.rover_traversal_graph = {r: {w: set() for w in self.waypoints} for r in self.rovers}
        self.visibility_graph = {w: set() for w in self.waypoints} # waypoint -> set(visible_waypoint)

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "at_lander":
                # Assuming only one lander
                self.lander_waypoint = parts[2]
            elif parts[0] == "equipped_for_soil_analysis":
                self.equipped_soil.add(parts[1])
            elif parts[0] == "equipped_for_rock_analysis":
                self.equipped_rock.add(parts[1])
            elif parts[0] == "equipped_for_imaging":
                self.equipped_imaging.add(parts[1])
            elif parts[0] == "store_of":
                self.store_of[parts[1]] = parts[2]
            elif parts[0] == "on_board":
                camera, rover = parts[1], parts[2] # Correct order from PDDL
                self.on_board.setdefault(rover, set()).add(camera)
            elif parts[0] == "supports":
                camera, mode = parts[1], parts[2]
                self.supports.setdefault(camera, set()).add(mode)
            elif parts[0] == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.calibration_target.setdefault(camera, set()).add(objective)
            elif parts[0] == "visible_from":
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from.setdefault(objective, set()).add(waypoint)
            elif parts[0] == "can_traverse":
                 rover, w1, w2 = parts[1], parts[2], parts[3]
                 if rover in self.rover_traversal_graph and w1 in self.waypoints and w2 in self.waypoints:
                    self.rover_traversal_graph[rover][w1].add(w2)
            elif parts[0] == "visible":
                 w1, w2 = parts[1], parts[2]
                 if w1 in self.waypoints and w2 in self.waypoints:
                    self.visibility_graph[w1].add(w2)


        # Precompute shortest path distances for each rover
        self.dist = {} # rover -> from_wp -> to_wp -> distance
        for rover in self.rovers:
            self.dist[rover] = {}
            # BFS from every waypoint
            for start_wp in self.waypoints:
                 # Ensure the rover has a traversal graph defined, otherwise it cannot move
                 if rover in self.rover_traversal_graph:
                     self.dist[rover][start_wp] = bfs(self.rover_traversal_graph[rover], start_wp)
                 else:
                     # If rover has no traversal graph, it cannot move anywhere except its start location
                     self.dist[rover][start_wp] = {start_wp: 0} # Only distance to itself is 0


        # Precompute lander-visible waypoints
        self.lander_visible_waypoints = set()
        if self.lander_waypoint and self.lander_waypoint in self.visibility_graph:
             for wp in self.waypoints:
                 if self.lander_waypoint in self.visibility_graph.get(wp, set()):
                     self.lander_visible_waypoints.add(wp)

        # Precompute shortest path distances from every waypoint to any lander-visible waypoint
        self.dist_to_lander = {} # rover -> from_wp -> distance
        for rover in self.rovers:
            self.dist_to_lander[rover] = {}
            for start_wp in self.waypoints:
                min_dist = float('inf')
                # Check if the rover's distance map from start_wp exists and is not just {start_wp: 0}
                if rover in self.dist and start_wp in self.dist[rover] and len(self.dist[rover][start_wp]) > 1:
                    for lander_wp in self.lander_visible_waypoints:
                        if lander_wp in self.dist[rover][start_wp]:
                             min_dist = min(min_dist, self.dist[rover][start_wp][lander_wp])
                elif rover in self.dist and start_wp in self.dist[rover] and len(self.dist[rover][start_wp]) == 1 and start_wp in self.lander_visible_waypoints:
                     # Rover cannot move, but is already at a lander-visible waypoint
                     min_dist = 0 # Distance to a lander-visible waypoint is 0
                # Otherwise, min_dist remains infinity

                self.dist_to_lander[rover][start_wp] = min_dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Helper to find rover location in the current state
        def get_rover_location(rover, current_state):
            for fact in current_state:
                if match(fact, "at", rover, "*"):
                    return get_parts(fact)[2]
            return None # Rover location not found in state (shouldn't happen in valid states)

        # Process each goal
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            goal_type = parts[0]

            if goal_type == "communicated_soil_data":
                waypoint = parts[1]
                min_goal_cost = float('inf')

                # Find a rover that can achieve this goal
                capable_rovers = [r for r in self.equipped_soil if r in self.rovers] # Ensure rover exists
                if not capable_rovers: return float('inf') # No rover can do this task

                for rover in capable_rovers:
                    rover_loc = get_rover_location(rover, state)
                    if rover_loc is None: continue # Should not happen

                    current_rover_cost = 0
                    sample_needed = True # Assume needed unless found in state
                    for fact in state:
                        if match(fact, "have_soil_analysis", rover, waypoint):
                            sample_needed = False
                            break

                    # Cost to get the sample if needed
                    if sample_needed:
                        current_rover_cost += 1 # sample_soil action
                        # Nav cost to sample location
                        nav_cost_to_sample = self.dist[rover][rover_loc].get(waypoint, float('inf'))
                        if nav_cost_to_sample == float('inf'): continue # Cannot reach sample location
                        current_rover_cost += nav_cost_to_sample
                        loc_after_data = waypoint # Rover is at sample location after sampling
                    else:
                        loc_after_data = rover_loc # Rover is at its current location if sample already exists

                    # Cost to communicate
                    current_rover_cost += 1 # communicate_soil_data action
                    # Nav cost from location after sample/have to lander-visible
                    nav_cost_to_lander = self.dist_to_lander[rover].get(loc_after_data, float('inf'))
                    if nav_cost_to_lander == float('inf'): continue # Cannot reach lander-visible location
                    current_rover_cost += nav_cost_to_lander

                    min_goal_cost = min(min_goal_cost, current_rover_cost)

                if min_goal_cost == float('inf'):
                     return float('inf') # Goal is unreachable
                total_cost += min_goal_cost

            elif goal_type == "communicated_rock_data":
                waypoint = parts[1]
                min_goal_cost = float('inf')

                # Find a rover that can achieve this goal
                capable_rovers = [r for r in self.equipped_rock if r in self.rovers] # Ensure rover exists
                if not capable_rovers: return float('inf') # No rover can do this task

                for rover in capable_rovers:
                    rover_loc = get_rover_location(rover, state)
                    if rover_loc is None: continue # Should not happen

                    current_rover_cost = 0
                    sample_needed = True # Assume needed unless found in state
                    for fact in state:
                        if match(fact, "have_rock_analysis", rover, waypoint):
                            sample_needed = False
                            break

                    # Cost to get the sample if needed
                    if sample_needed:
                        current_rover_cost += 1 # sample_rock action
                        # Nav cost to sample location
                        nav_cost_to_sample = self.dist[rover][rover_loc].get(waypoint, float('inf'))
                        if nav_cost_to_sample == float('inf'): continue # Cannot reach sample location
                        current_rover_cost += nav_cost_to_sample
                        loc_after_data = waypoint # Rover is at sample location after sampling
                    else:
                        loc_after_data = rover_loc # Rover is at its current location if sample already exists

                    # Cost to communicate
                    current_rover_cost += 1 # communicate_rock_data action
                    # Nav cost from location after sample/have to lander-visible
                    nav_cost_to_lander = self.dist_to_lander[rover].get(loc_after_data, float('inf'))
                    if nav_cost_to_lander == float('inf'): continue # Cannot reach lander-visible location
                    current_rover_cost += nav_cost_to_lander

                    min_goal_cost = min(min_goal_cost, current_rover_cost)

                if min_goal_cost == float('inf'):
                     return float('inf') # Goal is unreachable
                total_cost += min_goal_cost

            elif goal_type == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                min_goal_cost = float('inf')

                # Find a rover that can achieve this goal
                capable_rovers = [r for r in self.equipped_imaging if r in self.rovers] # Ensure rover exists
                if not capable_rovers: return float('inf') # No rover can do this task

                for rover in capable_rovers:
                    rover_loc = get_rover_location(rover, state)
                    if rover_loc is None: continue # Should not happen

                    current_rover_cost = 0
                    image_needed = True # Assume needed unless found in state
                    for fact in state:
                        if match(fact, "have_image", rover, objective, mode):
                            image_needed = False
                            break

                    # Cost to get the image if needed
                    if image_needed:
                        current_rover_cost += 1 # calibrate action
                        current_rover_cost += 1 # take_image action

                        # Nav cost to calibration waypoint
                        min_nav_to_cal = float('inf')
                        suitable_cal_wps = set()
                        # Find suitable cameras on this rover and their calibration waypoints
                        for camera in self.on_board.get(rover, set()):
                            if mode in self.supports.get(camera, set()):
                                for cal_target in self.calibration_target.get(camera, set()):
                                    cal_wps = self.objective_visible_from.get(cal_target, set())
                                    suitable_cal_wps.update(cal_wps)

                        if not suitable_cal_wps: continue # No suitable calibration waypoint for this rover/mode

                        for cal_wp in suitable_cal_wps:
                            nav_cost = self.dist[rover][rover_loc].get(cal_wp, float('inf'))
                            min_nav_to_cal = min(min_nav_to_cal, nav_cost)

                        if min_nav_to_cal == float('inf'): continue # Cannot reach any suitable calibration waypoint
                        current_rover_cost += min_nav_to_cal

                        # Nav cost from calibration waypoint to imaging waypoint
                        min_nav_cal_to_img = float('inf')
                        img_wps = self.objective_visible_from.get(objective, set())

                        if not img_wps: continue # No suitable imaging waypoint for this objective

                        # Need min dist from *any* suitable cal_wp (for any suitable camera/target on this rover)
                        # to *any* img_wp (for this objective/mode)
                        all_suitable_cal_wps_for_rover = set()
                        for camera in self.on_board.get(rover, set()):
                             if mode in self.supports.get(camera, set()):
                                 for cal_target in self.calibration_target.get(camera, set()):
                                     all_suitable_cal_wps_for_rover.update(self.objective_visible_from.get(cal_target, set()))

                        # Filter cal wps to only include those reachable from rover_loc
                        reachable_suitable_cal_wps = {
                            wp for wp in all_suitable_cal_wps_for_rover
                            if self.dist[rover][rover_loc].get(wp, float('inf')) != float('inf')
                        }

                        if not reachable_suitable_cal_wps: continue # Should be covered by min_nav_to_cal check, but double check

                        for cal_wp in reachable_suitable_cal_wps:
                            for img_wp in img_wps:
                                nav_cost = self.dist[rover][cal_wp].get(img_wp, float('inf'))
                                min_nav_cal_to_img = min(min_nav_cal_to_img, nav_cost)

                        if min_nav_cal_to_img == float('inf'): continue # Cannot navigate from cal to img waypoint
                        current_rover_cost += min_nav_cal_to_img

                        # Rover is at an imaging waypoint after taking the image
                        loc_after_data_is_img_wp = True
                        # Need min nav from *any* suitable img_wp to lander-visible
                        min_nav_img_to_lander = float('inf')
                        for img_wp in img_wps:
                             nav_cost = self.dist_to_lander[rover].get(img_wp, float('inf'))
                             min_nav_img_to_lander = min(min_nav_img_to_lander, nav_cost)
                        if min_nav_img_to_lander == float('inf'): continue # Cannot navigate from img wp to lander

                    else: # Image already exists for this rover
                        loc_after_data_is_img_wp = False # Rover is at its current location
                        min_nav_img_to_lander = float('inf') # Not needed in this branch, but initialize

                    # Cost to communicate
                    current_rover_cost += 1 # communicate_image_data action
                    if image_needed:
                         # Nav cost from imaging waypoint to lander-visible (precomputed min)
                         current_rover_cost += min_nav_img_to_lander
                    else:
                         # Nav cost from current location (r_loc) to lander-visible
                         nav_cost_to_lander = self.dist_to_lander[rover].get(rover_loc, float('inf'))
                         if nav_cost_to_lander == float('inf'): continue # Cannot reach lander-visible location
                         current_rover_cost += nav_cost_to_lander


                    min_goal_cost = min(min_goal_cost, current_rover_cost)

                if min_goal_cost == float('inf'):
                     return float('inf') # Goal is unreachable
                total_cost += min_goal_cost

        return total_cost
