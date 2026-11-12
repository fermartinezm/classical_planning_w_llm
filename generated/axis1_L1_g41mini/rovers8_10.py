from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check if a PDDL fact matches a given pattern with possible wildcards.

    Args:
        fact (str): A PDDL fact string like "(predicate arg1 arg2)".
        pattern (str): A sequence of strings or wildcards to match against fact parts.

    Returns:
        bool: True if the fact matches the pattern exactly; False otherwise.
    """
    parts = get_parts(fact)
    if len(parts) != len(pattern):
        return False
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (cdmmbhnd) domain.

    Summary:
    --------
    This heuristic estimates the number of actions required to achieve the goal
    by analyzing the goal requirements about imaging objectives in various qualities,
    the rovers' camera-status, and the rover-waypoint-objective relationships.
    It focuses on the minimum logical steps required to obtain the target image data 
    (lk059nof predicates) in the required qualities by rovers with cameras, 
    considering the need to position rovers at the correct waypoints, activate cameras
    if necessary, and complete image processing.

    Assumptions:
    ------------
    - Goals focus on `(lk059nof ?objective ?quality)` facts meaning an objective must be imaged
      in certain qualities (colour, high_res, low_res).
    - Cameras are attached to rovers, and each camera is associated with an objective.
    - Rovers can move between waypoints connected by `(kxvuge0a ?wp1 ?wp2)` and `(xuwncpn8 rover ?wp1 ?wp2)`.
    - Rovers must be at the right waypoint where the objective is located for imaging.
    - Cameras may need to be activated (v25mqvwj predicate) or calibrated (pmcrx641 vs zdfck87b).
    - Imaging involves some actions resulting in `l3943stc` and `lk059nof` predicates.
    - Facts about `chfbweyv` on waypoints indicate some processing or marking completed.
    - Some actions require rovers to hold a camera and be in the correct states.
    - Static predicates encode connectivity and capabilities.
    - The heuristic is non-admissible but aims to guide the search efficiently by focusing on missing target image facts,
      required rover positioning, camera activations, and essential intermediate states.

    Heuristic Initialization:
    -------------------------
    - Extract from static facts:
        * Waypoint connectivity graph to estimate shortest moves.
        * Camera-to-objective mappings.
        * Objective-to-waypoint mappings (via wecz86om).
        * Rovers with stores (v4e7u9dr) and their relation with rovers (fa5tam1q).
        * Movement relations (`xuwncpn8`).
    - Extract from goal conditions the set of `(lk059nof objective quality)` to satisfy.
    - Extract camera qualities available (xq9py0w8) to check available image modes.
    - Represent waypoints and rovers, objectives, qualities as sets for efficient queries.
    - Build a graph for waypoint adjacency to compute minimum moves.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Identify which `(lk059nof objective quality)` goals remain unsatisfied in the current state.
    2. For each unsatisfied goal:
       - Determine which rover(s) have a camera with the required quality for this objective.
       - Determine where the objective is located (via wecz86om).
       - Determine the current waypoint of the rover.
       - Calculate minimal movement cost (in terms of number of moves) for the rover to reach the objective's waypoint.
       - Check if the camera is currently activated for imaging (v25mqvwj).
         - If not, estimate the cost to activate/setup the camera (including store states pmcrx641/zdfck87b).
       - Check if imaging action effects like `l3943stc` exist; if not, estimate imaging action cost.
       - Sum movement, camera setup, imaging, and any intermediate actions needed.
    3. Also add costs for any required intermediate markers (e.g. chfbweyv on waypoints in goals).
    4. Sum these costs for all missing goal parts.
    5. Return zero heuristic if state meets the goal.
    6. Ensure estimates are finite by fallback defaults when information is missing.
    7. Aim for an efficient evaluation that scales with number of goals and rovers, avoiding heavy search/subgraph computations.

    This heuristic is designed to provide effective guidance by estimating logical "distinct action clusters":
    - Move rovers to objectives.
    - Activate camera and prepare data stores.
    - Capture images in required qualities.
    - Mark processing completion.

    It exploits static connectivity and relationships to approximate best cost to cover these requirements.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.operators = task.operators

        # Parse static info:

        # Build waypoint connectivity graph for shortest path moves:
        self.adj = {}
        for fact in self.static:
            if match(fact, "kxvuge0a", "*", "*"):
                _, wp1, wp2 = get_parts(fact)
                self.adj.setdefault(wp1, set()).add(wp2)
                self.adj.setdefault(wp2, set()).add(wp1)  # Assume undirected

        # Build movement edges specifically for rovers (xuwncpn8) to check rover-specific connectivity
        self.rover_moves = {}
        for fact in self.static:
            if match(fact, "xuwncpn8", "*", "*", "*"):
                _, rover, from_wp, to_wp = get_parts(fact)
                self.rover_moves.setdefault(rover, {})
                self.rover_moves[rover].setdefault(from_wp, set()).add(to_wp)

        # Map which cameras cover which objectives (g52pa0bi)
        self.camera_objectives = {}
        for fact in self.static:
            if match(fact, "g52pa0bi", "*", "*"):
                _, camera, objective = get_parts(fact)
                self.camera_objectives.setdefault(camera, set()).add(objective)

        # Map objective locations (wecz86om)
        self.objective_locations = {}
        for fact in self.static:
            if match(fact, "wecz86om", "*", "*"):
                _, obj, wp = get_parts(fact)
                self.objective_locations[obj] = wp

        # Map which cameras are attached to which rovers (uss6vmx6)
        self.camera_rovers = {}
        for fact in self.static:
            if match(fact, "uss6vmx6", "*", "*"):
                _, camera, rover = get_parts(fact)
                self.camera_rovers[camera] = rover

        # Map qualities available for each camera (xq9py0w8)
        self.camera_qualities = {}
        for fact in self.static:
            if match(fact, "xq9py0w8", "*", "*"):
                _, camera, quality = get_parts(fact)
                self.camera_qualities.setdefault(camera, set()).add(quality)

        # Store the rovers' stores (v4e7u9dr types) and maps to rovers (fa5tam1q)
        self.stores_for_rover = {}
        # Reverse map: store -> rover
        self.rover_for_store = {}
        for fact in self.static:
            if match(fact, "fa5tam1q", "*", "*"):
                _, store, rover = get_parts(fact)
                self.stores_for_rover.setdefault(rover, set()).add(store)
                self.rover_for_store[store] = rover

        # Pre-cache all rovers from static facts about rovers (dp3yzgup), (vg9nzfll), (jy67mgmv)
        self.rovers = set()
        for fact in self.static:
            if match(fact, "dp3yzgup", "*"):
                _, rover = get_parts(fact)
                self.rovers.add(rover)
            elif match(fact, "vg9nzfll", "*"):
                _, rover = get_parts(fact)
                self.rovers.add(rover)
            elif match(fact, "jy67mgmv", "*"):
                _, rover = get_parts(fact)
                self.rovers.add(rover)

        # Predicates relevant for rover positioning: ilkg22gp rover waypoint
        # We will find rovers' current positions from state.

        # Precompute distances function later using BFS.

    def _shortest_path(self, start, goal, rover=None):
        """
        Compute the shortest path length for a rover from start waypoint to goal waypoint.

        If rover is given and has specific movement constraints (xuwncpn8),
        use that adjacency, otherwise fallback to general waypoint adjacency.

        Returns integer length or a high number if no path found.
        """
        if start == goal:
            return 0

        if rover and rover in self.rover_moves:
            adj = self.rover_moves[rover]
        else:
            adj = self.adj

        queue = [(start, 0)]
        visited = set([start])

        while queue:
            current, dist = queue.pop(0)
            for nbr in adj.get(current, set()):
                if nbr == goal:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        # No path found; return large cost
        return 50

    def __call__(self, node):
        state = node.state

        # Quick goal check: if state meets goals, heuristic = 0
        if self.goals <= state:
            return 0

        # --- Extract relevant facts from current state ---

        # 1) Get currently achieved lk059nof goals from state
        achieved_lk059nof = set()
        for fact in state:
            if match(fact, "lk059nof", "*", "*"):
                _, objective, quality = get_parts(fact)
                achieved_lk059nof.add((objective, quality))

        # 2) Get chfbweyv facts (marker goals)
        achieved_chfbweyv = set()
        for fact in state:
            if match(fact, "chfbweyv", "*"):
                _, wp = get_parts(fact)
                achieved_chfbweyv.add(wp)

        # 3) Get rovers' current positions (ilkg22gp rover waypoint)
        rover_positions = {}
        for fact in state:
            if match(fact, "ilkg22gp", "*", "*"):
                _, rover, wp = get_parts(fact)
                rover_positions[rover] = wp

        # 4) Which cameras are currently activated on which rovers? (v25mqvwj camera rover)
        camera_activated = set()
        for fact in state:
            if match(fact, "v25mqvwj", "*", "*"):
                _, camera, rover = get_parts(fact)
                camera_activated.add((camera, rover))

        # 5) pmcrx641 indicates roverstore status; zdfck87b indicates store activated
        store_status_pmcrx641 = set()  # stores in state with pmcrx641 status (inactive?)
        store_status_zdfck87b = set()  # stores in state with zdfck87b status (active?)

        for fact in state:
            if match(fact, "pmcrx641", "*"):
                _, store = get_parts(fact)
                store_status_pmcrx641.add(store)
            elif match(fact, "zdfck87b", "*"):
                _, store = get_parts(fact)
                store_status_zdfck87b.add(store)

        # 6) Store which stores belong to which rover (reverse map self.rover_for_store)

        # 7) l3943stc predicates indicate imaging captured in state (l3943stc rover objective quality)
        captured_images = set()
        for fact in state:
            if match(fact, "l3943stc", "*", "*", "*"):
                _, rover, objective, quality = get_parts(fact)
                captured_images.add((rover, objective, quality))

        # --- Calculate heuristic cost ---

        total_cost = 0

        # Cost parameters: estimate number of actions for each logical step
        MOVE_COST = 1          # one move action between waypoints
        ACTIVATE_STORE_COST = 2  # approx cost to activate store: cp8otkvs + o64yr71v
        ACTIVATE_CAMERA_COST = 2 # approximate cost for camera activation (actions wyhd5xzi)
        IMAGE_CAPTURE_COST = 1   # cost to capture image (jx1xpj7f or tkek73dx)
        MARK_PROCESSING_COST = 1 # cost to mark waypoint processed (a7xj6iq1 or n84tdn2k)
        IMAGE_PROCESS_COST = 1   # cost for final goal imaging predicate (bypa1zm7)

        # 1) Estimate missing lk059nof goals:
        for goal_fact in self.goals:
            if not match(goal_fact, "lk059nof", "*", "*"):
                continue
            _, obj, qual = get_parts(goal_fact)
            if (obj, qual) in achieved_lk059nof:
                continue  # Already done goal imaging

            # Find cameras with this quality and objective
            candidate_cameras = []
            for camera, objs in self.camera_objectives.items():
                if obj in objs:
                    if qual in self.camera_qualities.get(camera, set()):
                        candidate_cameras.append(camera)

            if not candidate_cameras:
                # No camera supports this quality for this objective; assign large cost
                total_cost += 50
                continue

            # For each candidate camera find rover and cost to position rover + activate camera + take image
            best_cost = 50  # large number for minimal selection

            for camera in candidate_cameras:
                rover = self.camera_rovers.get(camera)
                if rover is None or rover not in rover_positions:
                    # Unknown rover or position, assign large cost
                    continue
                rover_wp = rover_positions[rover]

                obj_wp = self.objective_locations.get(obj)
                if obj_wp is None:
                    # Unknown objective location assign penalty
                    obj_wp = rover_wp  # fallback no move

                # Movement cost: shortest path by rover moves (fallback: general waypoint adjacency)
                move_cost = self._shortest_path(rover_wp, obj_wp, rover=rover) * MOVE_COST

                # Determine if store activation needed:
                stores = self.stores_for_rover.get(rover, set())
                store_activation_cost = 0
                # If any store belonging to rover is inactive (pmcrx641) and no active store (zdfck87b), count activation cost
                has_active_store = any(store in store_status_zdfck87b for store in stores)
                has_inactive_store = any(store in store_status_pmcrx641 for store in stores)
                if has_inactive_store and not has_active_store:
                    store_activation_cost += ACTIVATE_STORE_COST

                # Determine if camera is already activated
                cam_activated = (camera, rover) in camera_activated
                camera_activation_cost = 0 if cam_activated else ACTIVATE_CAMERA_COST

                # Check if image already captured by this rover in this quality for this objective
                imaged_already = (rover, obj, qual) in captured_images
                image_capture_cost = 0 if imaged_already else IMAGE_CAPTURE_COST

                cost = move_cost + store_activation_cost + camera_activation_cost + image_capture_cost

                if cost < best_cost:
                    best_cost = cost

            total_cost += best_cost

        # 2) Estimate chfbweyv waypoint goals missing processing marks
        for goal_fact in self.goals:
            if match(goal_fact, "chfbweyv", "*"):
                _, waypoint = get_parts(goal_fact)
                if waypoint not in achieved_chfbweyv:
                    # Minimal estimate: one or two marking actions plus moving a rover there
                    # Find any rover closest to waypoint
                    min_move = 50
                    for rover in self.rovers:
                        if rover not in rover_positions:
                            continue
                        move_dist = self._shortest_path(rover_positions[rover], waypoint, rover=rover)
                        if move_dist < min_move:
                            min_move = move_dist
                    total_cost += min_move * MOVE_COST + MARK_PROCESSING_COST

        # 3) Estimate final goal predicates for lk059nof if any not directly implied by existing l3943stc
        # They tend to require some processing by bypa1zm7 action, presumed to cost IMAGE_PROCESS_COST
        # But we included imaging cost above for l3943stc; if lk059nof predicate not yet true, add this cost
        for goal_fact in self.goals:
            if not match(goal_fact, "lk059nof", "*", "*"):
                continue
            _, obj, qual = get_parts(goal_fact)
            if (obj, qual) in achieved_lk059nof:
                continue
            # The l3943stc predicate (image capture) might be done, check
            captured_any_rover = any(
                (rover, obj, qual) in captured_images for rover in self.rovers if rover in rover_positions
            )
            if captured_any_rover:
                total_cost += IMAGE_PROCESS_COST

        # Heuristic cannot be negative
        cost = max(total_cost, 1)

        return cost
