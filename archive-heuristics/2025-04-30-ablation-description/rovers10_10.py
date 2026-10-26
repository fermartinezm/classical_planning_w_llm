from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string.
    - `args`: The expected pattern (wildcards `*` allowed).
    """
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

# The roversHeuristic class
class roversHeuristic: # Assuming inheritance from Heuristic base class is handled externally or not needed for this specific output
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up the estimated costs
    for each unachieved goal predicate. The estimated cost for a single goal predicate
    involves travel costs (approximated by shortest path distances) and action costs
    (e.g., sample, take_image, communicate). It considers whether the necessary data/image
    has already been acquired by a rover.

    # Assumptions
    - The primary goals are communicating soil data, rock data, and image data.
    - Travel cost between waypoints is 1 per step (edge in the can_traverse graph).
    - Action costs (sample, take_image, communicate) are 1.
    - Calibration and dropping samples are not explicitly modeled in the heuristic cost,
      assuming they are either negligible or handled locally by the search.
    - When data/image is not yet acquired, the cost includes travel to the acquisition
      location (sample waypoint or imaging waypoint) and then travel to a communication
      waypoint.
    - When data/image is acquired by a rover, the cost includes travel from that rover's
      current location to a communication waypoint.
    - For travel costs from a fixed waypoint (like a sample location or imaging location)
      to a communication waypoint, the shortest path is calculated using the graph of
      a representative rover (the first one found).
    - Reachability is checked using precomputed distances; unreachable goals result in
      infinite heuristic cost.

    # Heuristic Initialization
    - Parses static facts to identify:
        - Lander locations.
        - Waypoint connectivity for each rover (`can_traverse`).
        - Waypoint visibility (`visible`).
        - Rover equipment (`equipped_for_soil_analysis`, etc.).
        - Camera properties (`on_board`, `supports`, `calibration_target`).
        - Objective visibility from waypoints (`visible_from`).
    - Identifies communication waypoints (visible from lander locations).
    - Builds a waypoint graph for each rover based on `can_traverse` facts.
    - Precomputes all-pairs shortest path distances for each rover using BFS.
    - Stores goal predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Get the current state facts and current locations of all rovers.
    3. For each goal predicate in the task's goals:
        a. If the goal predicate is already true in the current state, add 0 cost for this goal and continue to the next goal.
        b. If the goal is `(communicated_soil_data ?w)`:
            i. Check if any soil-equipped rover currently has `(have_soil_analysis ?r ?w)`.
            ii. If no rover has the data:
                - Check if `(at_soil_sample ?w)` is in the state. If not, assume the sample was taken but data is not held (cost calculation proceeds from waypoint `?w`). If yes, calculate the minimum travel cost for any soil-equipped rover from its current location to `?w`. Add this travel cost + 1 (for `sample_soil` action) to the total.
                - Calculate the minimum travel cost from waypoint `?w` to any communication waypoint (using a representative rover's graph). Add this travel cost + 1 (for `communicate_soil_data` action) to the total.
            iii. If a rover `?r` has the data:
                - Calculate the minimum travel cost for rover `?r` from its current location to any communication waypoint. Add this travel cost + 1 (for `communicate_soil_data` action) to the total.
            iv. If any required travel is impossible, return infinity.
        c. If the goal is `(communicated_rock_data ?w)`: Follow similar logic as for soil data, using rock-equipped rovers and `have_rock_analysis`.
        d. If the goal is `(communicated_image_data ?o ?m)`:
            i. Check if any imaging-equipped rover with a suitable camera currently has `(have_image ?r ?o ?m)`.
            ii. If no rover has the image:
                - Identify suitable imaging waypoints (`visible_from ?o ?p`). If none exist, return infinity.
                - Identify suitable imaging rovers (equipped for imaging, with camera supporting mode `?m`). If none exist, return infinity.
                - Calculate the minimum travel cost for any suitable imaging rover from its current location to any suitable imaging waypoint. Add this travel cost + 1 (for `take_image` action) to the total. (Calibration cost is ignored).
                - Calculate the minimum travel cost from any suitable imaging waypoint to any communication waypoint (using a representative rover's graph). Add this travel cost + 1 (for `communicate_image_data` action) to the total.
            iii. If a rover `?r` has the image:
                - Calculate the minimum travel cost for rover `?r` from its current location to any communication waypoint. Add this travel cost + 1 (for `communicate_image_data` action) to the total.
            iv. If any required travel is impossible, return infinity.
    4. If the total cost is 0, it means all goal predicates were already in the state, which implies the state is the goal state. Return 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # Assuming task object has attributes: goals, static, operators, initial_state, facts, name
        # super().__init__(task) # Call the base class constructor if inheriting

        self.goals = task.goals # Goal conditions.
        static_facts = task.static # Facts that are not affected by actions.

        # --- Parse Static Facts ---
        self.lander_locations = set()
        self.can_traverse = {} # rover -> set of (wp1, wp2)
        self.visible = set() # (wp1, wp2)
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.store_of = {} # store -> rover
        self.on_board = {} # camera -> rover
        self.supports = {} # camera -> set of modes
        self.calibration_target = {} # camera -> objective
        self.visible_from = {} # objective -> set of waypoints
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.landers = set()
        self.stores = set()

        # Collect all potential objects and waypoints from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            args = parts[1:]

            if pred == "at_lander":
                if len(args) == 2:
                    self.landers.add(args[0])
                    self.lander_locations.add(args[1])
                    self.waypoints.add(args[1])
            elif pred == "can_traverse":
                if len(args) == 3:
                    rover, w1, w2 = args
                    self.rovers.add(rover)
                    self.waypoints.add(w1)
                    self.waypoints.add(w2)
                    if rover not in self.can_traverse:
                        self.can_traverse[rover] = set()
                    self.can_traverse[rover].add((w1, w2))
                    self.can_traverse[rover].add((w2, w1)) # Assuming undirected graph
            elif pred == "visible":
                if len(args) == 2:
                    w1, w2 = args
                    self.waypoints.add(w1)
                    self.waypoints.add(w2)
                    self.visible.add((w1, w2))
                    self.visible.add((w2, w1)) # Assuming symmetric visibility
            elif pred == "equipped_for_soil_analysis":
                if len(args) == 1:
                    self.equipped_soil.add(args[0])
                    self.rovers.add(args[0])
            elif pred == "equipped_for_rock_analysis":
                if len(args) == 1:
                    self.equipped_rock.add(args[0])
                    self.rovers.add(args[0])
            elif pred == "equipped_for_imaging":
                if len(args) == 1:
                    self.equipped_imaging.add(args[0])
                    self.rovers.add(args[0])
            elif pred == "store_of":
                if len(args) == 2:
                    self.stores.add(args[0])
                    self.rovers.add(args[1])
                    self.store_of[args[0]] = args[1]
            elif pred == "on_board":
                if len(args) == 2:
                    self.cameras.add(args[0])
                    self.rovers.add(args[1])
                    self.on_board[args[0]] = args[1]
            elif pred == "supports":
                if len(args) == 2:
                    self.cameras.add(args[0])
                    self.modes.add(args[1])
                    if args[0] not in self.supports:
                        self.supports[args[0]] = set()
                    self.supports[args[0]].add(args[1])
            elif pred == "calibration_target":
                if len(args) == 2:
                    self.cameras.add(args[0])
                    self.objectives.add(args[1])
                    self.calibration_target[args[0]] = args[1]
            elif pred == "visible_from":
                if len(args) == 2:
                    self.objectives.add(args[0])
                    self.waypoints.add(args[1])
                    if args[0] not in self.visible_from:
                        self.visible_from[args[0]] = set()
                    self.visible_from[args[0]].add(args[1])
            # Also collect waypoints mentioned in static facts about samples or rover initial positions
            elif pred in ("at_soil_sample", "at_rock_sample"):
                 if len(args) == 1:
                     self.waypoints.add(args[0])
            elif pred == "at" and len(args) == 2 and args[0].startswith("rover"):
                 self.waypoints.add(args[1])


        # Identify communication waypoints (visible from any lander location)
        self.comm_waypoints = set()
        for lander_loc in self.lander_locations:
            for w1, w2 in self.visible:
                if w2 == lander_loc:
                    self.comm_waypoints.add(w1)

        # --- Precompute Distances ---
        self.distances = {} # rover -> start_wp -> end_wp -> distance
        for rover in self.rovers:
            graph = {}
            # Ensure all known waypoints are in the graph, even if not connected by this rover
            for wp in self.waypoints:
                graph[wp] = set()
            if rover in self.can_traverse:
                for w1, w2 in self.can_traverse[rover]:
                    # Add edge only if both waypoints are known
                    if w1 in graph and w2 in graph:
                         graph[w1].add(w2)
                         graph[w2].add(w1) # Assuming undirected
            self.distances[rover] = {}
            for start_wp in self.waypoints:
                self.distances[rover][start_wp] = self._bfs(graph, start_wp)

        # Choose a representative rover for general waypoint distances if needed
        self.representative_rover = next(iter(self.rovers), None)


    def _bfs(self, graph, start_node):
        """Perform BFS to find distances from start_node to all other nodes."""
        distances = {node: float('inf') for node in graph}
        if start_node not in graph:
             # Start node is not in the graph of known waypoints for this rover
             return distances # All distances remain infinity

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            for neighbor in graph.get(current_node, set()):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def get_distance(self, rover, start_wp, end_wp):
        """Get the precomputed distance between two waypoints for a specific rover."""
        if start_wp is None or end_wp is None:
             return float('inf') # Cannot calculate distance if waypoint is None

        # If rover is specified, use its graph
        if rover is not None and rover in self.distances:
            if start_wp in self.distances[rover] and end_wp in self.distances[rover][start_wp]:
                 return self.distances[rover][start_wp][end_wp]
            # If waypoints are not in the graph for this rover, they are unreachable
            return float('inf')

        # If rover is None, use the representative rover's graph
        if self.representative_rover and self.representative_rover in self.distances:
             if start_wp in self.distances[self.representative_rover] and end_wp in self.distances[self.representative_rover][start_wp]:
                 return self.distances[self.representative_rover][start_wp][end_wp]
             # If waypoints are not in the graph for the representative rover, they are unreachable
             return float('inf')

        # No rovers or graphs available
        return float('inf')


    def is_goal(self, state):
        """Check if the current state satisfies all goal conditions."""
        return self.goals <= state


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state # Current world state.

        # If the goal is already reached, the heuristic is 0.
        if self.is_goal(state):
            return 0

        total_cost = 0

        # Get current rover locations
        current_rover_locations = {}
        for fact in state:
            if match(fact, "at", "rover*", "*"):
                rover, wp = get_parts(fact)[1:]
                current_rover_locations[rover] = wp

        # Iterate through each goal predicate
        for goal in self.goals:
            # If this goal predicate is already true, it contributes 0 to the heuristic
            if goal in state:
                continue

            # Parse the goal predicate
            parts = get_parts(goal)
            if not parts or len(parts) < 2: continue # Skip malformed goals

            pred = parts[0]

            if pred == "communicated_soil_data":
                if len(parts) < 2: continue # Malformed goal
                waypoint = parts[1]

                # Check if any rover already has the soil analysis data for this waypoint
                rover_holding_data = next((r for r in self.equipped_soil if f"(have_soil_analysis {r} {waypoint})" in state), None)

                if rover_holding_data:
                    # Rover has data, need to communicate from its current location
                    min_dist_to_comm = float('inf')
                    if rover_holding_data in current_rover_locations:
                        start_wp = current_rover_locations[rover_holding_data]
                        for comm_wp in self.comm_waypoints:
                            dist = self.get_distance(rover_holding_data, start_wp, comm_wp)
                            if dist is not None:
                                min_dist_to_comm = min(min_dist_to_comm, dist)
                    else:
                         # Rover holding data is not at a waypoint? Invalid state?
                         return float('inf') # Indicate unsolvable path

                    if min_dist_to_comm == float('inf'): return float('inf') # Cannot reach comm point
                    total_cost += min_dist_to_comm + 1 # Travel + Communicate action

                else:
                    # No rover has the data, need to sample and communicate
                    sample_available = f"(at_soil_sample {waypoint})" in state

                    if sample_available:
                        # Sample is available, need to go sample it first
                        min_dist_to_sample = float('inf')
                        if self.equipped_soil:
                            for rover in self.equipped_soil:
                                if rover in current_rover_locations:
                                    dist = self.get_distance(rover, current_rover_locations[rover], waypoint)
                                    if dist is not None:
                                        min_dist_to_sample = min(min_dist_to_sample, dist)
                        else: return float('inf') # No equipped rover

                        if min_dist_to_sample == float('inf'): return float('inf') # Cannot reach sample
                        total_cost += min_dist_to_sample + 1 # Travel + Sample action

                    # Need to communicate from the sample waypoint (or where it was)
                    start_wp_for_comm = waypoint
                    min_dist_to_comm = float('inf')
                    # Use a representative rover's graph for general waypoint distances
                    if self.representative_rover:
                        for comm_wp in self.comm_waypoints:
                            dist = self.get_distance(self.representative_rover, start_wp_for_comm, comm_wp)
                            if dist is not None:
                                min_dist_to_comm = min(min_dist_to_comm, dist)
                    else: return float('inf') # No rovers available

                    if min_dist_to_comm == float('inf'): return float('inf') # Cannot reach comm point
                    total_cost += min_dist_to_comm + 1 # Travel + Communicate action

            elif pred == "communicated_rock_data":
                if len(parts) < 2: continue # Malformed goal
                waypoint = parts[1]

                # Check if any rover already has the rock analysis data for this waypoint
                rover_holding_data = next((r for r in self.equipped_rock if f"(have_rock_analysis {r} {waypoint})" in state), None)

                if rover_holding_data:
                    # Rover has data, need to communicate from its current location
                    min_dist_to_comm = float('inf')
                    if rover_holding_data in current_rover_locations:
                        start_wp = current_rover_locations[rover_holding_data]
                        for comm_wp in self.comm_waypoints:
                            dist = self.get_distance(rover_holding_data, start_wp, comm_wp)
                            if dist is not None:
                                min_dist_to_comm = min(min_dist_to_comm, dist)
                    else: return float('inf') # Invalid state?

                    if min_dist_to_comm == float('inf'): return float('inf') # Cannot reach comm point
                    total_cost += min_dist_to_comm + 1 # Travel + Communicate action

                else:
                    # No rover has the data, need to sample and communicate
                    sample_available = f"(at_rock_sample {waypoint})" in state

                    if sample_available:
                        # Sample is available, need to go sample it first
                        min_dist_to_sample = float('inf')
                        if self.equipped_rock:
                            for rover in self.equipped_rock:
                                if rover in current_rover_locations:
                                    dist = self.get_distance(rover, current_rover_locations[rover], waypoint)
                                    if dist is not None:
                                        min_dist_to_sample = min(min_dist_to_sample, dist)
                        else: return float('inf') # No equipped rover

                        if min_dist_to_sample == float('inf'): return float('inf') # Cannot reach sample
                        total_cost += min_dist_to_sample + 1 # Travel + Sample action

                    # Need to communicate from the sample waypoint (or where it was)
                    start_wp_for_comm = waypoint
                    min_dist_to_comm = float('inf')
                    # Use a representative rover's graph for general waypoint distances
                    if self.representative_rover:
                        for comm_wp in self.comm_waypoints:
                            dist = self.get_distance(self.representative_rover, start_wp_for_comm, comm_wp)
                            if dist is not None:
                                min_dist_to_comm = min(min_dist_to_comm, dist)
                    else: return float('inf') # No rovers available

                    if min_dist_to_comm == float('inf'): return float('inf') # Cannot reach comm point
                    total_cost += min_dist_to_comm + 1 # Travel + Communicate action


            elif pred == "communicated_image_data":
                if len(parts) < 3: continue # Malformed image goal
                objective = parts[1]
                mode = parts[2]

                # Check if any rover already has the image
                rover_holding_image = next((r for r in self.equipped_imaging if f"(have_image {r} {objective} {mode})" in state), None)

                # Find suitable imaging rovers (equipped for imaging, with camera supporting mode)
                R_img_suitable = set()
                for rover in self.equipped_imaging:
                    # Check if this rover has any camera that supports the required mode
                    has_suitable_camera = False
                    for camera, rov in self.on_board.items():
                        if rov == rover and mode in self.supports.get(camera, set()):
                            has_suitable_camera = True
                            break
                    if has_suitable_camera:
                        R_img_suitable.add(rover)

                # Find suitable imaging waypoints (visible from objective)
                W_img_suitable = self.visible_from.get(objective, set())


                if rover_holding_image:
                    # Rover has image, need to communicate from its current location
                    min_dist_to_comm = float('inf')
                    if rover_holding_image in current_rover_locations:
                        start_wp = current_rover_locations[rover_holding_image]
                        for comm_wp in self.comm_waypoints:
                            dist = self.get_distance(rover_holding_image, start_wp, comm_wp)
                            if dist is not None:
                                min_dist_to_comm = min(min_dist_to_comm, dist)
                    else: return float('inf') # Invalid state?

                    if min_dist_to_comm == float('inf'): return float('inf') # Cannot reach comm point
                    total_cost += min_dist_to_comm + 1 # Travel + Communicate action

                else:
                    # No rover has image, need to image and communicate
                    if not R_img_suitable or not W_img_suitable:
                         return float('inf') # Cannot take image

                    # Cost to image: min dist from suitable rover to suitable imaging waypoint
                    min_dist_to_image_wp = float('inf')
                    for rover in R_img_suitable:
                        if rover in current_rover_locations:
                            start_wp = current_rover_locations[rover]
                            for img_wp in W_img_suitable:
                                dist = self.get_distance(rover, start_wp, img_wp)
                                if dist is not None:
                                    min_dist_to_image_wp = min(min_dist_to_image_wp, dist)
                    if min_dist_to_image_wp == float('inf'): return float('inf') # Cannot reach imaging location
                    total_cost += min_dist_to_image_wp + 1 # Travel + Take image action (ignoring calibration)

                    # Need to communicate from a suitable imaging waypoint
                    min_dist_image_wp_to_comm = float('inf')
                    # Use a representative rover's graph for general waypoint distances
                    if self.representative_rover and self.comm_waypoints:
                        for img_wp in W_img_suitable:
                            for comm_wp in self.comm_waypoints:
                                dist = self.get_distance(self.representative_rover, img_wp, comm_wp)
                                if dist is not None:
                                    min_dist_image_wp_to_comm = min(min_dist_image_wp_to_comm, dist)
                        min_dist_to_comm = min_dist_image_wp_to_comm
                    else: return float('inf') # No rovers or comm points

                    if min_dist_to_comm == float('inf'): return float('inf') # Cannot reach comm point
                    total_cost += min_dist_to_comm + 1 # Travel + Communicate action


        # If total_cost is 0, it means all goal predicates were already in the state.
        # Since the goal only contains 'communicated_...' predicates, total_cost == 0
        # implies self.is_goal(state) is true.
        # We return 0 if the goal is reached, and total_cost > 0 otherwise.
        # The check `if self.is_goal(state): return 0` at the beginning handles the goal state.
        # If we reach here and total_cost is still 0, it means the initial check was skipped
        # but all goals were somehow covered by the loop logic (which shouldn't happen if
        # the initial check is present).
        # Let's ensure we return 0 only for the actual goal state.
        # If total_cost > 0, return total_cost. If total_cost == 0, it must be the goal state.
        # So simply returning total_cost is correct given the initial check.

        return total_cost
