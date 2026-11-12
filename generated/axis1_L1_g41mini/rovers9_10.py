from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(ilkg22gp rover1 waypoint2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (cdmmbhnd) domain.

    Summary:
        This heuristic estimates the number of remaining actions needed to achieve 
        the goal conditions related to taking layered measurements (lk059nof predicates) 
        of objectives with different resolutions (colour, high_res, low_res) using rovers, cameras, and waypoints.

    Assumptions:
        - Goals consist mainly of (lk059nof objective resolution) facts.
        - To achieve these goals, rovers must:
            * Move between waypoints.
            * Have cameras take measurements at objectives.
            * Potentially prepare rovers' internal states and stores before capturing images.
        - Movement between waypoints is only possible if (xuwncpn8 rover from to) and (ilkg22gp rover from) hold.
        - Camera must be assigned to rover and objective (v25mqvwj camera rover).
        - Capturing an image involves certain preconditions and effects reflected in predicates like:
          (l3943stc rover objective resolution), which correlate with (lk059nof objective resolution).
        - The static facts provide the connectivity graph between waypoints (kxvuge0a facts) and possible rover moves (xuwncpn8).

    Heuristic Initialization:
        - Extract all goal (lk059nof) predicates with their associated objectives and resolutions.
        - From static facts, build:
            * Bidirectional waypoint connectivity graph for rover travel (from kxvuge0a).
            * Maps of rovers' possible moves between waypoints (xuwncpn8).
            * Cameras' assignments to objectives (g52pa0bi).
            * Camera types' supported resolutions (xq9py0w8).
        - Extract rovers' stores (v4e7u9dr), their states (pmcrx641 and zdfck87b), and rovers themselves (cdmmbhnd).
    
    Step-By-Step Thinking for Computing Heuristic:
        1. Identify which goals have already been satisfied in the current state: 
           those with (lk059nof objective resolution) present require zero cost.
        2. For each unsatisfied goal (lk059nof), find:
           - The objective and resolution required.
           - The rover currently assigned or closest to the objective.
           - Whether the corresponding camera with the correct resolution is assigned to the rover.
        3. Estimate the cost to:
           a) Move the rover to the waypoint containing the objective (if not there), using shortest path over waypoints.
           b) Prepare the rover if needed (e.g., stores ready).
           c) Capture the measurement at the objective with the camera.
        4. Sum estimated costs over all unsatisfied goals.
        5. Add a penalty for rovers not at correct waypoints when measurement needed.
        6. Return 0 if all goals are satisfied.

    Efficiency considerations:
        - Use precomputed shortest paths for rover waypoint movement.
        - Estimate cost in action counts: moving one waypoint = 1 action, capturing measurement = 1-2 actions,
          preparing the rover store = 1-2 actions.
        - Avoid expensive computations, focus on reachable objectives.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goal objectives with needed resolutions: set of (objective, resolution)
        self.goal_lk059nof = set()
        for fact in self.goals:
            if match(fact, "lk059nof", "*", "*"):
                parts = get_parts(fact)
                self.goal_lk059nof.add((parts[1], parts[2]))

        # Extract waypoints from static facts (from kxvuge0a predicate)
        self.waypoint_graph = {}
        for fact in self.static:
            if match(fact, "kxvuge0a", "*", "*"):
                _, wp1, wp2 = get_parts(fact)
                # Build bidirectional graph
                if wp1 not in self.waypoint_graph:
                    self.waypoint_graph[wp1] = set()
                self.waypoint_graph[wp1].add(wp2)

        # Extract rover moves connectivity (xuwncpn8 rover from to)
        # Map: rover -> dict of from_wp -> set(to_wps)
        self.rover_move_graph = {}
        for fact in self.static:
            if match(fact, "xuwncpn8", "*", "*", "*"):
                _, rover, from_wp, to_wp = get_parts(fact)
                if rover not in self.rover_move_graph:
                    self.rover_move_graph[rover] = {}
                if from_wp not in self.rover_move_graph[rover]:
                    self.rover_move_graph[rover][from_wp] = set()
                self.rover_move_graph[rover][from_wp].add(to_wp)

        # Extract which cameras are assigned to which objectives (g52pa0bi camera objective)
        self.camera_objectives = {}
        for fact in self.static:
            if match(fact, "g52pa0bi", "*", "*"):
                _, camera, objective = get_parts(fact)
                if camera not in self.camera_objectives:
                    self.camera_objectives[camera] = set()
                self.camera_objectives[camera].add(objective)

        # Extract what resolutions cameras support (xq9py0w8 camera resolution)
        self.camera_resolutions = {}
        for fact in self.static:
            if match(fact, "xq9py0w8", "*", "*"):
                _, camera, resolution = get_parts(fact)
                if camera not in self.camera_resolutions:
                    self.camera_resolutions[camera] = set()
                self.camera_resolutions[camera].add(resolution)

        # Extract rovers (cdmmbhnd type) from static (dp3yzgup ?wgox9 or jy67mgmv ?wgox9 indicate rover presence)
        self.rovers = set()
        for fact in self.static:
            if match(fact, "dp3yzgup", "*") or match(fact, "jy67mgmv", "*"):
                parts = get_parts(fact)
                self.rovers.add(parts[1])

        # Extract rovers' stores (v4e7u9dr) and links between rover <-> store (fa5tam1q store rover)
        self.store_of_rover = {}
        for fact in self.static:
            if match(fact, "fa5tam1q", "*", "*"):
                _, store, rover = get_parts(fact)
                self.store_of_rover[rover] = store

        # Since shortest path calculation between waypoints is frequent, prepare
        # a generic shortest path helper method for rover move graphs

    def shortest_path_length(self, rover, start, goal):
        """
        Compute shortest path length for rover moving from start to goal waypoint,
        using the edges in rover_move_graph which reflect possible moves (xuwncpn8).
        BFS is used since all moves cost 1.
        Returns a large number if no path exists.
        """
        if start == goal:
            return 0

        graph = self.rover_move_graph.get(rover, {})
        visited = set()
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            if current == goal:
                return dist
            if current in visited:
                continue
            visited.add(current)
            for nxt in graph.get(current, []):
                if nxt not in visited:
                    queue.append((nxt, dist + 1))
        # Return large number for unreachable
        return 1000

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to achieve all missing goals.

        Heuristic estimation components:
          - Rover moves to objectives waypoints.
          - Preparing rover's store if needed.
          - Taking measurements with cameras.
          - Assigning cameras to rovers if required (simplified or assumed as zero here).
        """
        state = node.state

        # Quick goal check: if all goal facts satisfied, heuristic is zero.
        if self.goal_lk059nof <= {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "lk059nof", "*", "*")}:
            return 0

        # Facts indexed by predicate for efficiency
        ilkg22gp_facts = {get_parts(f)[1:3] for f in state if match(f, "ilkg22gp", "*", "*")}  # (rover, waypoint) rover location
        v25mqvwj_facts = {(get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "v25mqvwj", "*", "*")}  # (camera, rover) camera assigned to rover
        l3943stc_facts = {(get_parts(f)[1], get_parts(f)[2], get_parts(f)[3]) for f in state if match(f, "l3943stc", "*", "*", "*")}  # (rover, objective, resolution) measurement done by rover
        pmcrx641_facts = {get_parts(f)[1] for f in state if match(f, "pmcrx641", "*")}  # roverstore ready or not
        zdfck87b_facts = {get_parts(f)[1] for f in state if match(f, "zdfck87b", "*")}  # roverstore alternative state

        fa5tam1q_facts = {get_parts(f) for f in state if match(f, "fa5tam1q", "*", "*")}  # (store, rover)

        # Map rover to its current waypoint
        rover_locations = {}
        for rover, wp in ilkg22gp_facts:
            if rover in self.rovers:
                rover_locations[rover] = wp

        # Map store states from state (pmcrx641 => ready?, zdfck87b => alternative)
        rovers_stores_ready = {}
        for rover in self.rovers:
            store = self.store_of_rover.get(rover)
            if not store:
                continue
            # If store's pmcrx641 fact present, count as ready state
            if store in pmcrx641_facts:
                rovers_stores_ready[rover] = True
            elif store in zdfck87b_facts:
                rovers_stores_ready[rover] = False
            else:
                rovers_stores_ready[rover] = False  # default no

        # Identify cameras and their assigned rovers
        camera_rover_map = {}
        for (camera, rover) in v25mqvwj_facts:
            camera_rover_map[camera] = rover

        # For each unsatisfied goal (objective, resolution), estimate cost to achieve
        total_estimated_cost = 0

        for (obj, res) in self.goal_lk059nof:
            if (obj, res) in {(o, r) for (_, o, r) in l3943stc_facts} or (obj, res) in {(get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "lk059nof", "*", "*")}:
                # Already completed measurement by rover or goal lk059nof present
                continue

            # Find rovers that can do the measurement (have camera assigned with support for that resolution and objective)
            # We heuristically check cameras assigned to rovers that can image this objective and resolution
            candidate_rovers = []
            for camera, assigned_rover in camera_rover_map.items():
                if assigned_rover not in self.rovers:
                    continue
                # camera supports resolution?
                if camera in self.camera_resolutions and res in self.camera_resolutions[camera]:
                    # camera can image objective?
                    if camera in self.camera_objectives and obj in self.camera_objectives[camera]:
                        candidate_rovers.append(assigned_rover)

            if not candidate_rovers:
                # No camera-rover combination matches resolution/objective: penalty cost to assign (set to 3)
                total_estimated_cost += 3
                # As fallback, skip detailed estimation for this goal.
                continue

            # Choose rover with minimal move cost
            best_cost_for_goal = 1000  # large initial cost

            # Get objective's relevant waypoint(s) from state or static - must infer the waypoint of objective
            # The (wecz86om objective waypoint) relations map objective to waypoint(s).
            objective_waypoints = set()
            for fact in self.static:
                if match(fact, "wecz86om", obj, "*"):
                    objective_waypoints.add(get_parts(fact)[2])
            # Also try from current state in case of dynamic assignment (safer)
            for fact in state:
                if match(fact, "wecz86om", obj, "*"):
                    objective_waypoints.add(get_parts(fact)[2])

            if not objective_waypoints:
                # If no waypoint known, penalize
                total_estimated_cost += 10
                continue

            for rover in candidate_rovers:
                rover_wp = rover_locations.get(rover)
                if rover_wp is None:
                    # Unknown rover location, add penalty
                    move_cost = 5
                else:
                    # shortest path from rover_wp to any objective waypoint
                    move_cost = min(
                        self.shortest_path_length(rover, rover_wp, wp) for wp in objective_waypoints
                    )

                # Check if rover store ready, else add +1 action for preparation
                prep_cost = 0 if rovers_stores_ready.get(rover, False) else 1

                # Cost to take measurement (assumed 1 action)
                measure_cost = 1

                # Total cost for this rover option
                cost = move_cost + prep_cost + measure_cost

                if cost < best_cost_for_goal:
                    best_cost_for_goal = cost

            total_estimated_cost += best_cost_for_goal

        # Ensure heuristic is never zero except for goal states
        if total_estimated_cost == 0:
            return 1

        return total_estimated_cost
