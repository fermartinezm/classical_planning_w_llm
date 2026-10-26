from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or invalid format gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards `*` allowed.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(start_node, graph, all_nodes):
    """
    Performs Breadth-First Search to find shortest distances from a start node
    in a graph represented as an adjacency dictionary.
    """
    distances = {node: float('inf') for node in all_nodes}

    if start_node not in all_nodes:
         # Start node is not part of the graph nodes, cannot reach anything
         return distances

    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()
        # Check if current_node exists in graph keys before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It calculates the minimum estimated cost for each
    unmet goal independently and sums these minimum costs. The estimated cost
    for a single goal is the sum of estimated navigation steps and action costs
    (sample/image, calibrate, communicate, drop) required to achieve that specific
    goal fact, assuming the necessary resources (rovers, cameras, samples,
    communication links) are available. Navigation cost is estimated using
    precomputed shortest paths for each rover.

    # Assumptions
    - The heuristic calculates the cost for each goal independently and sums them.
      This ignores potential synergies (e.g., one navigation reaching waypoints
      for multiple goals) or conflicts (e.g., multiple rovers needing the same
      resource).
    - Navigation cost between waypoints for a specific rover is the shortest
      path distance in the graph traversable by that rover.
    - Samples (`at_soil_sample`, `at_rock_sample`) only disappear when collected.
      If a sample is not present at a waypoint and no rover has the analysis,
      that specific soil/rock goal is considered impossible via sampling that waypoint.
    - Calibration is consumed by taking an image. Recalibration is needed for
      subsequent images with the same camera/rover.
    - If a required goal is deemed impossible to achieve from the current state
      (e.g., sample gone, no suitable rover/camera/waypoint combination), the
      heuristic returns infinity.

    # Heuristic Initialization
    The constructor precomputes static information from the task definition:
    - Navigation graphs for each rover based on `can_traverse` and `visible` facts.
      Shortest path distances between all pairs of waypoints are computed using BFS.
    - Locations of landers.
    - Waypoints visible from lander locations (communication points).
    - Rover capabilities (`equipped_for_soil_analysis`, etc.).
    - Store assignments for each rover.
    - Imaging-related facts: `visible_from` for objectives, `calibration_target`
      for cameras, `on_board` cameras for rovers, `supports` modes for cameras.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state `s`:
    1. Initialize `total_cost = 0`.
    2. Extract relevant dynamic facts from `s`: current rover locations,
       `have_soil_analysis`, `have_rock_analysis`, `have_image`, `calibrated`,
       `at_soil_sample`, `at_rock_sample`, `full` stores, and `communicated_X_data` goals achieved.
    3. For each goal fact `g` in `task.goals`:
        a. If `g` is already in the set of achieved goals in state `s`, continue (cost is 0 for this goal).
        b. If `g` is `(communicated_soil_data ?w)`:
            i. Find suitable rovers (equipped for soil analysis). If none, this goal is impossible (min_goal_cost = inf).
            ii. For each suitable rover `r`:
                - Get its current location `r_loc`.
                - Find its store `s_store`.
                - Calculate `rover_cost`:
                    - If `(have_soil_analysis r w)` is in state `s`: Cost is `min_x(dist(r, r_loc, x) + 1)` over communication waypoints `x`.
                    - Else (`(have_soil_analysis r w)` not in state `s`):
                        - If `(at_soil_sample w)` is in state `s`:
                            - Sample cost = `dist(r, r_loc, w) + 1`. Add 1 if `(full s_store)` is in state `s` (for drop).
                            - Communicate cost = `min_x(dist(r, w, x) + 1)` over communication waypoints `x`.
                            - Total cost = Sample cost + Communicate cost.
                        - Else (`(at_soil_sample w)` not in state `s`): Cost is infinity for this rover path (sample is gone).
                - Update `min_goal_cost = min(min_goal_cost, rover_cost)`.
            iii. If `min_goal_cost` is still infinity, return `float('inf')` (problem likely unsolvable).
            iv. Add `min_goal_cost` to `total_cost`.
        c. If `g` is `(communicated_rock_data ?w)`: Follow similar logic as soil data.
        d. If `g` is `(communicated_image_data ?o ?m)`:
            i. Find suitable rover/camera pairs (`r` equipped for imaging, `i` on board `r` supporting `m`). If none, this goal is impossible (min_goal_cost = inf).
            ii. For each suitable pair `r`, `i`:
                - Get `r_loc`.
                - Find image waypoints `p` for `o`. If none, ignore this pair.
                - Find calibration target `t` for `i`. If none, ignore this pair.
                - Find calibration waypoints `w` for `t`. If none, ignore this pair.
                - Calculate `pair_cost`:
                    - If `(have_image r o m)` is in state `s`: Cost is `min_x(dist(r, r_loc, x) + 1)` over communication waypoints `x`.
                    - Else (`(have_image r o m)` not in state `s`):
                        - Calculate `image_comm_cost = float('inf')` over image waypoints `p`:
                            - Calculate `cal_image_cost`:
                                - If `(calibrated i r)` is in state `s`: Cost is `dist(r, r_loc, p) + 1` (take_image).
                                - Else (`(calibrated i r)` not in state `s`):
                                    - Find best calibration waypoint `w_best` minimizing `dist(r, r_loc, w) + 1` over calibration waypoints `w`. If no reachable calibration waypoint, `cal_image_cost` is infinity.
                                    - If reachable: Calibration cost = `dist(r, r_loc, w_best) + 1`.
                                    - Image cost after calibration = `dist(r, w_best, p) + 1`.
                                    - Total `cal_image_cost` = Calibration cost + Image cost after calibration.
                                    # Note: If best_cal_wp is None or dist is inf, cal_image_cost remains inf.

                            # If image can be taken (cal_image_cost is finite)
                            if cal_image_cost != float('inf'):
                                # Cost to communicate after taking image (rover is at image_wp)
                                comm_cost = float('inf')
                                for comm_wp in self.comm_waypoints:
                                    dist = self.rover_nav_graphs.get(rover, {}).get(image_wp, {}).get(comm_wp, float('inf'))
                                    if dist != float('inf'):
                                        comm_cost = min(comm_cost, dist + 1) # +1 for communicate action

                                if comm_cost != float('inf'):
                                    image_comm_cost = min(image_comm_cost, cal_image_cost + comm_cost)
                            # Else: Cannot communicate after taking image. image_comm_cost remains inf.

                            pair_cost = image_comm_cost # Minimum cost over all image_wps

                        min_goal_cost = min(min_goal_cost, pair_cost)

            # If after checking all options for this goal, the cost is still infinity,
            # it means this specific goal is unreachable from the current state.
            if min_goal_cost == float('inf'):
                 # If any required goal is impossible, the whole state is likely unsolvable.
                 # Return infinity.
                 return float('inf')

            total_cost += min_goal_cost

        return total_cost

    def dist(self, rover, from_wp, to_wp):
        """Helper to get precomputed distance, returns inf if not found."""
        return self.rover_nav_graphs.get(rover, {}).get(from_wp, {}).get(to_wp, float('inf'))


    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Collect all waypoints mentioned in static or initial facts ---
        all_waypoints = set()
        all_rovers = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate in ["can_traverse", "visible"]:
                if len(parts) >= 3:
                    all_waypoints.add(parts[1])
                    all_waypoints.add(parts[2])
            elif predicate in ["at", "at_lander", "visible_from"]:
                 if len(parts) >= 3: # (at rover wp), (at_lander lander wp), (visible_from obj wp)
                     all_waypoints.add(parts[2])
                 elif len(parts) >= 2 and predicate == "at_soil_sample": # (at_soil_sample wp)
                     all_waypoints.add(parts[1])
                 elif len(parts) >= 2 and predicate == "at_rock_sample": # (at_rock_sample wp)
                     all_waypoints.add(parts[1])
            elif predicate == "at": # (at ?x - rover ?y - waypoint) in static?
                 if len(parts) >= 3 and parts[1].startswith('rover'): # Simple check for rover type
                     all_rovers.add(parts[1])
                     all_waypoints.add(parts[2])


        for fact in initial_state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate in ["at", "at_lander", "visible_from"]:
                 if len(parts) >= 3: # (at rover wp), (at_lander lander wp), (visible_from obj wp)
                     all_waypoints.add(parts[2])
                 elif len(parts) >= 2 and predicate == "at_soil_sample": # (at_soil_sample wp)
                     all_waypoints.add(parts[1])
                 elif len(parts) >= 2 and predicate == "at_rock_sample": # (at_rock_sample wp)
                     all_waypoints.add(parts[1])
            elif predicate == "at": # (at ?x - rover ?y - waypoint) in initial state
                 if len(parts) >= 3 and parts[1].startswith('rover'): # Simple check for rover type
                     all_rovers.add(parts[1])
                     all_waypoints.add(parts[2])


        # --- Build Rover Navigation Graphs ---
        self.rover_nav_graphs = {} # rover -> {from_wp -> {to_wp -> distance}}

        # Build base graph from visible facts (symmetric)
        visible_graph = collections.defaultdict(set)
        for fact in static_facts:
             if match(fact, "visible", "*", "*"):
                 y, z = get_parts(fact)[1], get_parts(fact)[2]
                 visible_graph[y].add(z)
                 visible_graph[z].add(y) # Visible is symmetric

        # Build rover-specific graphs based on can_traverse and visible
        for rover in all_rovers:
            rover_graph = collections.defaultdict(set)
            for fact in static_facts:
                 if match(fact, "can_traverse", rover, "*", "*"):
                     r, y, z = get_parts(fact)
                     # Edge y->z exists if (can_traverse r y z) AND (visible y z)
                     if z in visible_graph.get(y, set()):
                         rover_graph[y].add(z)

            # Compute all-pairs shortest paths using BFS from each waypoint
            self.rover_nav_graphs[rover] = {}
            # Use the collected set of all waypoints for initialization
            for start_wp in all_waypoints:
                self.rover_nav_graphs[rover][start_wp] = bfs(start_wp, rover_graph, all_waypoints)


        # --- Extract other Static and Initial Facts ---
        self.lander_locations = set()
        self.comm_waypoints = set() # Waypoints visible from any lander location
        self.rover_capabilities = collections.defaultdict(set) # rover -> {capability}
        self.rover_stores = {} # rover -> store
        self.camera_calibration_target = {} # camera -> objective
        self.objective_image_waypoints = collections.defaultdict(set) # objective -> {waypoint}
        self.calibration_target_waypoints = collections.defaultdict(set) # objective (target) -> {waypoint}
        self.rover_cameras = collections.defaultdict(set) # rover -> {camera}
        self.camera_supported_modes = collections.defaultdict(set) # camera -> {mode}

        # Collect facts from static and initial state that are static over time
        # (i.e., not effects of any action)
        # In this domain, most facts are dynamic except for capabilities, store_of,
        # calibration_target, on_board, supports, visible, can_traverse, at_lander, visible_from.
        # Let's process static facts first.
        static_and_initial_facts = set(static_facts) | set(initial_state) # Include initial state for things like initial rover/lander pos, initial samples etc.

        lander_at_facts = set()
        visible_facts_set = set() # Use a different name to avoid conflict with the graph

        for fact in static_and_initial_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]

            if predicate == "at_lander":
                lander, loc = parts[1], parts[2]
                self.lander_locations.add(loc)
                lander_at_facts.add(fact) # Keep track to find comm waypoints later
            elif predicate == "equipped_for_soil_analysis":
                if len(parts) >= 2: self.rover_capabilities[parts[1]].add('soil')
            elif predicate == "equipped_for_rock_analysis":
                if len(parts) >= 2: self.rover_capabilities[parts[1]].add('rock')
            elif predicate == "equipped_for_imaging":
                if len(parts) >= 2: self.rover_capabilities[parts[1]].add('imaging')
            elif predicate == "store_of":
                if len(parts) >= 3: self.rover_stores[parts[2]] = parts[1] # rover -> store
            elif predicate == "calibration_target":
                if len(parts) >= 3: self.camera_calibration_target[parts[1]] = parts[2]
            elif predicate == "on_board":
                if len(parts) >= 3: self.rover_cameras[parts[2]].add(parts[1])
            elif predicate == "supports":
                if len(parts) >= 3: self.camera_supported_modes[parts[1]].add(parts[2])
            elif predicate == "visible_from":
                if len(parts) >= 3:
                    obj, wp = parts[1], parts[2]
                    self.objective_image_waypoints[obj].add(wp)
            elif predicate == "visible":
                 visible_facts_set.add(fact)


        # Determine communication waypoints: visible from any lander location
        for lander_loc in self.lander_locations:
             for fact in visible_facts_set:
                 y, z = get_parts(fact)[1], get_parts(fact)[2]
                 if y == lander_loc:
                     self.comm_waypoints.add(z)
                 elif z == lander_loc:
                     self.comm_waypoints.add(y)

        # Populate calibration_target_waypoints based on camera_calibration_target and objective_image_waypoints
        for camera, cal_target_obj in self.camera_calibration_target.items():
             if cal_target_obj in self.objective_image_waypoints:
                 self.calibration_target_waypoints[cal_target_obj].update(self.objective_image_waypoints[cal_target_obj])

