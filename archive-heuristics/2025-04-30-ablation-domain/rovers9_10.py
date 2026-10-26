from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in this path
# If the base class is in a different location, adjust the import path accordingly.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy Heuristic class if the import fails, for standalone testing
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not found")
        def __repr__(self):
            return "<DummyHeuristic>"


# Helper functions to parse PDDL facts
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
    # Check if the number of parts matches the number of args, considering wildcards
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to find shortest paths
def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest distances from a start node
    to all reachable nodes in a graph.

    Args:
        graph: Adjacency list representation of the graph (dict: node -> list of neighbors).
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its shortest distance from start_node.
        Returns {start_node: 0} if start_node is not in graph keys.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    if start_node not in graph:
         # Start node is not in the graph's keys (no outgoing edges)
         # It might still be a valid waypoint, just isolated for this rover.
         # The distance to itself is 0. Distances to others remain unknown (implicitly infinity).
         return distances

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is a valid key in the graph before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    goal conditions. It sums the estimated costs for each unachieved goal
    independently. The cost for each goal includes steps to acquire necessary
    data (sample or image) and steps to move a capable rover to the lander
    for communication.

    # Assumptions
    - Each unachieved goal contributes independently to the heuristic.
    - Resource constraints like rover store capacity and camera calibration
      persistence are simplified or ignored for cost estimation (e.g., assumes
      an empty store is available when needed, assumes calibration might need
      repetition but calculates minimum path considering recalibration).
    - Movement cost between waypoints is the shortest path distance for the
      specific rover.
    - Action costs are uniform (1 per action).
    - The problem is solvable (finite paths exist between necessary locations).

    # Heuristic Initialization
    - Parses static facts to identify lander location, rover capabilities,
      camera properties (on-board, supports, calibration target), objective
      visibility, sample locations, and rover-specific traverse graphs.
    - Parses goal facts to identify required communications (soil, rock, image).
    - Precomputes shortest path distances between all pairs of waypoints
      for each rover using BFS on their respective traverse graphs.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all goal conditions that are not met in the current state.
    2. Initialize the total heuristic cost with the number of unachieved goals.
       This accounts for the final communication action for each goal.
    3. For each unachieved soil data goal `(communicated_soil_data ?w)`:
       - Find the minimum additional cost among all soil-equipped rovers to:
         a. Achieve the prerequisite `(have_soil_analysis ?r ?w)`. If already true, cost is 0.
            If not true and a soil sample exists at `?w`, the cost is the distance
            from the rover's current location to `?w` plus 1 (for the `take_soil_sample` action).
            (Simplified: ignores the need for an empty store). The rover is assumed
            to be at `?w` after taking the sample, or at its current location if
            analysis was already done.
         b. Move the rover from the location where the analysis prerequisite was met
            (or its current location if already met) to the lander's location.
            The cost is the shortest path distance.
       - Add the minimum combined cost (prerequisite actions + movement to lander)
         found over all suitable rovers to the total heuristic cost.
    4. For each unachieved rock data goal `(communicated_rock_data ?w)`:
       - Follow a similar logic as for soil data goals, using rock-specific
         predicates and sample locations.
    5. For each unachieved image data goal `(communicated_image_data ?o ?m)`:
       - Find the minimum additional cost among all imaging-equipped rovers
         with cameras supporting mode `?m` and calibrated for objective `?o` to:
         a. Achieve the prerequisite `(have_image ?r ?o ?m)`. If already true, cost is 0.
            If not true, the cost involves calibration and taking the image.
            This requires moving to a waypoint `?w_cal` visible from `?o` to calibrate
            (cost: distance + 1), and then moving to a waypoint `?w_img` visible from `?o`
            to take the image (cost: distance + 1). If the camera is already calibrated,
            the calibration step is skipped. The minimum path cost involving these steps
            is calculated over all suitable `?w_cal` and `?w_img` visible from `?o`.
            The rover is assumed to be at `?w_img` after taking the image, or at its
            current location if the image was already taken.
         b. Move the rover from the location where the image prerequisite was met
            (or its current location if already met) to the lander's location.
            The cost is the shortest path distance.
       - Add the minimum combined cost (prerequisite actions + movement to lander)
         found over all suitable rover/camera combinations to the total heuristic cost.
    6. Return the total calculated cost. If any necessary movement or prerequisite
       step is impossible (e.g., no path exists), that specific rover/camera
       option is ignored by using infinity, and the minimum will reflect only
       reachable options. If all options are impossible, the base cost (number
       of unachieved goals) remains, indicating a potentially difficult or
       unsolvable subproblem.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Collect all waypoints mentioned in static facts to build graphs
        all_waypoints = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["at_lander", "at", "visible", "can_traverse", "at_soil_sample", "at_rock_sample", "visible_from"]:
                 for part in parts[1:]:
                     # Waypoints are typically objects starting with 'waypoint'
                     if part.startswith("waypoint"):
                         all_waypoints.add(part)

        self.lander_location = None
        self.rover_capabilities = {} # rover -> set of capabilities (e.g., 'soil', 'rock', 'imaging')
        self.store_to_rover = {} # store -> rover
        self.rover_cameras = {} # rover -> list of cameras
        self.camera_modes = {} # camera -> set of modes
        self.camera_calibration_targets = {} # camera -> objective
        self.objective_visible_from = {} # objective -> set of waypoints
        self.soil_sample_locations = set() # waypoint
        self.rock_sample_locations = set() # waypoint
        self.all_rovers = set() # Collect all rover names

        # Rover-specific traverse graphs and precomputed distances
        self.rover_waypoint_graphs = {} # rover -> dict (waypoint -> list of neighbors)
        self.rover_distances = {} # rover -> dict (start_wp -> dict (end_wp -> distance))

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at_lander":
                self.lander_location = parts[2]
            elif predicate.startswith("equipped_for_"):
                capability = predicate.split("_")[2] # soil, rock, imaging
                rover = parts[1]
                self.all_rovers.add(rover)
                self.rover_capabilities.setdefault(rover, set()).add(capability)
            elif predicate == "store_of":
                store, rover = parts[1], parts[2]
                self.store_to_rover[store] = rover
            elif predicate == "can_traverse":
                rover, w1, w2 = parts[1], parts[2], parts[3]
                self.all_rovers.add(rover)
                self.rover_waypoint_graphs.setdefault(rover, {}).setdefault(w1, []).append(w2)
                all_waypoints.add(w1) # Ensure waypoints from can_traverse are included
                all_waypoints.add(w2)
            elif predicate == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.camera_calibration_targets[camera] = objective
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.all_rovers.add(rover)
                self.rover_cameras.setdefault(rover, []).append(camera)
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif predicate == "at_soil_sample":
                self.soil_sample_locations.add(parts[1])
                all_waypoints.add(parts[1])
            elif predicate == "at_rock_sample":
                self.rock_sample_locations.add(parts[1])
                all_waypoints.add(parts[1])
            elif predicate == "visible_from":
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from.setdefault(objective, set()).add(waypoint)
                all_waypoints.add(waypoint)

        # Ensure lander location is in the set of all waypoints
        if self.lander_location:
             all_waypoints.add(self.lander_location)

        # Precompute distances for each rover
        for rover in self.all_rovers:
            graph = self.rover_waypoint_graphs.get(rover, {})
            self.rover_distances[rover] = {}
            # Run BFS from every waypoint that exists in the graph for this rover
            # or any waypoint mentioned in the problem (as start/end points)
            for start_wp in all_waypoints:
                 # Only run BFS if the start_wp is actually in the rover's graph keys
                 # This avoids running BFS from isolated waypoints if the graph is sparse
                 if start_wp in graph:
                    self.rover_distances[rover][start_wp] = bfs(graph, start_wp)
                 else:
                     # If start_wp is not in graph keys, it might still be a valid waypoint
                     # but the rover cannot move *from* it using can_traverse facts.
                     # BFS from here would only find distance 0 to itself.
                     # We can represent this as {start_wp: 0}
                     self.rover_distances[rover][start_wp] = {start_wp: 0}


        # Parse goal facts
        self.goal_soil_waypoints = set()
        self.goal_rock_waypoints = set()
        self.goal_image_objectives_modes = set() # set of (objective, mode) tuples

        # Goals can be complex structures, need to flatten ANDs
        def flatten_goals(goal_structure):
            goals = []
            if isinstance(goal_structure, tuple) and goal_structure[0] == 'and':
                for sub_goal in goal_structure[1:]:
                    goals.extend(flatten_goals(sub_goal))
            else:
                goals.append(goal_structure)
            return goals

        flat_goals = flatten_goals(task.goals)

        for goal in flat_goals:
            # Goal facts are represented as strings in the state, so parse the goal structure
            # which might be a tuple representation from the PDDL parser.
            # Convert goal tuples like ('communicated_soil_data', 'waypoint1') to string '(communicated_soil_data waypoint1)'
            if isinstance(goal, tuple):
                 goal_str = "(" + " ".join(goal) + ")"
            elif isinstance(goal, str):
                 goal_str = goal # Already a string fact
            else:
                 continue # Ignore other types

            parts = get_parts(goal_str)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                self.goal_soil_waypoints.add(parts[1])
            elif predicate == "communicated_rock_data":
                self.goal_rock_waypoints.add(parts[1])
            elif predicate == "communicated_image_data":
                self.goal_image_objectives_modes.add((parts[1], parts[2]))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract relevant dynamic facts from the current state
        rover_at = {} # rover -> waypoint
        current_have_soil = set() # (rover, waypoint)
        current_have_rock = set() # (rover, waypoint)
        current_have_image = set() # (rover, objective, mode)
        current_calibrated = set() # (camera, rover)
        # current_empty_stores = set() # store # Simplified: ignore store for heuristic cost

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at":
                rover_at[parts[1]] = parts[2]
            elif predicate == "have_soil_analysis":
                current_have_soil.add((parts[1], parts[2]))
            elif predicate == "have_rock_analysis":
                current_have_rock.add((parts[1], parts[2]))
            elif predicate == "have_image":
                current_have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == "calibrated":
                current_calibrated.add((parts[1], parts[2]))
            # elif predicate == "empty":
            #     current_empty_stores.add(parts[1])

        # Identify unachieved goals
        unachieved_soil = {w for w in self.goal_soil_waypoints if f'(communicated_soil_data {w})' not in state}
        unachieved_rock = {w for w in self.goal_rock_waypoints if f'(communicated_rock_data {w})' not in state}
        unachieved_image = {(o, m) for (o, m) in self.goal_image_objectives_modes if f'(communicated_image_data {o} {m})' not in state}

        # If all goals are achieved, heuristic is 0
        if not (unachieved_soil or unachieved_rock or unachieved_image):
            return 0

        # Base cost: 1 action (communicate) for each unachieved goal
        total_cost = len(unachieved_soil) + len(unachieved_rock) + len(unachieved_image)

        # Estimate additional costs for each unachieved goal
        infinity = float('inf') # Use infinity for unreachable costs

        # Soil Goals
        for soil_waypoint in unachieved_soil:
            min_prereq_move_cost = infinity
            # Iterate over rovers that exist and are equipped
            for rover in self.all_rovers:
                 if 'soil' in self.rover_capabilities.get(rover, set()):
                    current_w = rover_at.get(rover)
                    if current_w is None: continue # Rover location unknown

                    cost_to_get_analysis = infinity
                    location_after_analysis = None

                    if (rover, soil_waypoint) in current_have_soil:
                        cost_to_get_analysis = 0
                        location_after_analysis = current_w
                    elif soil_waypoint in self.soil_sample_locations:
                         # Cost to move to sample location and take sample
                         # Need distance from current_w to soil_waypoint
                         dist_to_sample = self.rover_distances.get(rover, {}).get(current_w, {}).get(soil_waypoint, infinity)

                         if dist_to_sample is not infinity:
                             cost_to_get_analysis = dist_to_sample + 1 # Move + Take
                             location_after_analysis = soil_waypoint

                    if cost_to_get_analysis is not infinity and location_after_analysis is not None:
                         # Cost to move from analysis location to lander
                         # Need distance from location_after_analysis to lander_location
                         dist_to_lander = self.rover_distances.get(rover, {}).get(location_after_analysis, {}).get(self.lander_location, infinity)
                         if dist_to_lander is not infinity:
                             cost_prereq_move = cost_to_get_analysis + dist_to_lander
                             min_prereq_move_cost = min(min_prereq_move_cost, cost_prereq_move)

            if min_prereq_move_cost is not infinity:
                total_cost += min_prereq_move_cost
            # else: goal is unreachable by any equipped rover, base cost (1) remains

        # Rock Goals
        for rock_waypoint in unachieved_rock:
            min_prereq_move_cost = infinity
            # Iterate over rovers that exist and are equipped
            for rover in self.all_rovers:
                 if 'rock' in self.rover_capabilities.get(rover, set()):
                    current_w = rover_at.get(rover)
                    if current_w is None: continue

                    cost_to_get_analysis = infinity
                    location_after_analysis = None

                    if (rover, rock_waypoint) in current_have_rock:
                        cost_to_get_analysis = 0
                        location_after_analysis = current_w
                    elif rock_waypoint in self.rock_sample_locations:
                         # Cost to move to sample location and take sample
                         dist_to_sample = self.rover_distances.get(rover, {}).get(current_w, {}).get(rock_waypoint, infinity)
                         if dist_to_sample is not infinity:
                             cost_to_get_analysis = dist_to_sample + 1 # Move + Take
                             location_after_analysis = rock_waypoint

                    if cost_to_get_analysis is not infinity and location_after_analysis is not None:
                         # Cost to move from analysis location to lander
                         dist_to_lander = self.rover_distances.get(rover, {}).get(location_after_analysis, {}).get(self.lander_location, infinity)
                         if dist_to_lander is not infinity:
                             cost_prereq_move = cost_to_get_analysis + dist_to_lander
                             min_prereq_move_cost = min(min_prereq_move_cost, cost_prereq_move)

            if min_prereq_move_cost is not infinity:
                total_cost += min_prereq_move_cost
            # else: goal is unreachable

        # Image Goals
        for objective, mode in unachieved_image:
            min_prereq_move_cost = infinity
            w_visible = self.objective_visible_from.get(objective, set())
            if not w_visible: continue # Cannot image this objective if not visible from anywhere

            # Iterate over rovers that exist and are equipped
            for rover in self.all_rovers:
                 if 'imaging' in self.rover_capabilities.get(rover, set()):
                    current_w = rover_at.get(rover)
                    if current_w is None: continue

                    for camera in self.rover_cameras.get(rover, []):
                        if mode in self.camera_modes.get(camera, set()) and \
                           self.camera_calibration_targets.get(camera) == objective: # Suitable camera

                            cost_to_get_image = infinity
                            location_after_image = None

                            if (rover, objective, mode) in current_have_image:
                                cost_to_get_image = 0
                                location_after_image = current_w
                            else:
                                # Need to take image
                                min_image_path_cost = infinity
                                best_w_img = None # Need this to calculate communication cost

                                for w_img in w_visible:
                                    dist_curr_to_img = self.rover_distances.get(rover, {}).get(current_w, {}).get(w_img, infinity)
                                    if dist_curr_to_img is infinity: continue

                                    if (camera, rover) in current_calibrated:
                                        # Already calibrated
                                        path_cost = dist_curr_to_img + 1 # Move + Take
                                        if path_cost < min_image_path_cost:
                                            min_image_path_cost = path_cost
                                            best_w_img = w_img
                                    else:
                                        # Need to calibrate first
                                        for w_cal in w_visible: # Assuming calibration needs visible waypoint
                                            dist_curr_to_cal = self.rover_distances.get(rover, {}).get(current_w, {}).get(w_cal, infinity)
                                            dist_cal_to_img = self.rover_distances.get(rover, {}).get(w_cal, {}).get(w_img, infinity)
                                            if dist_curr_to_cal is infinity or dist_cal_to_img is infinity: continue

                                            path_cost = dist_curr_to_cal + 1 # Move + Calibrate
                                            path_cost += dist_cal_to_img + 1 # Move + Take
                                            if path_cost < min_image_path_cost:
                                                min_image_path_cost = path_cost
                                                best_w_img = w_img # Store the w_img that gave the min path

                                if min_image_path_cost is not infinity:
                                    cost_to_get_image = min_image_path_cost
                                    location_after_image = best_w_img

                            if cost_to_get_image is not infinity and location_after_image is not None:
                                 # Cost to move from image location to lander
                                 dist_to_lander = self.rover_distances.get(rover, {}).get(location_after_image, {}).get(self.lander_location, infinity)
                                 if dist_to_lander is not infinity:
                                     cost_prereq_move = cost_to_get_image + dist_to_lander
                                     min_prereq_move_cost = min(min_prereq_move_cost, cost_prereq_move)

            if min_prereq_move_cost is not infinity:
                total_cost += min_prereq_move_cost
            # else: goal is unreachable

        return total_cost
