from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(ilkg22gp rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to achieve all goal facts
    of the form (lk059nof ?objective ?colour), i.e., that each objective has all required
    image types (e.g. colour, low_res, high_res). The heuristic approximates:
    - How many images of required types are still missing,
    - The number of rovers needed to capture those images,
    - Movement cost for rovers to reach objectives,
    - The status of image stores (whether recharged and usable),
    - Whether cameras are currently mounted or need to be mounted.

    The heuristic value is an estimate of the minimal number of actions required to:
    1) Move rovers to waypoints near objectives,
    2) Mount cameras and set their modes,
    3) Capture required images for objectives,
    4) Recharge stores if depleted,
    5) Upload images to stores.

    Assumptions
    -----------
    - Goals require taking images (lk059nof) of multiple qualities for several objectives.
    - Each rover can move along connected waypoints.
    - Rovers need a mounted camera to capture images.
    - Orders of actions roughly correspond to moving, mounting, capturing, and uploading.
    - The heuristic ignores some subtle dependencies and uses approximations.
    - Movement cost between waypoints is approximated by shortest path counts in the static graph.
    - Cameras and stores are modeled by binary state inferences from predicates.
    - Heuristic is not admissible but focuses on guiding greedy search efficiently.

    Heuristic Initialization
    ------------------------
    - Extract goal objectives and required image types from goal facts.
    - Extract static movement graph between waypoints from static facts (kxvuge0a predicates).
    - Extract mapping from objectives to waypoints (wecz86om).
    - Extract which cameras and stores exist, and which rovers have access to which stores.
    - Store sets of rovers and waypoints (from static facts).
    - Precompute waypoint adjacency as a dictionary for shortest-path computations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a current state:

    1. Identify which required images for each objective are already achieved 
       by checking goal image facts (lk059nof).

    2. For each missing image type on each objective, estimate:
       - The closest rover that can take it (based on rover current positions and movement graph).
       - The cost for the rover to move to the waypoint near the objective.
       - Whether the rover has a mounted camera with the correct mode or must mount/change it.
       - Whether the rover's store is recharged, or if reset action (o64yr71v) is needed beforehand.

    3. Sum the minimal estimated costs for each missing image capture:
       - Movement cost (shortest path length),
       - Mounting camera/setup cost (fixed cost),
       - Capture cost (fixed cost per image),
       - Upload cost (fixed cost if needed),
       - Store recharge cost if required.

    4. Return the total sum as heuristic estimate.

    This approach encourages the search to focus on the nearest rovers with ready equipment,
    minimizing expansion by giving informative yet computationally efficient guidance.
    """

    # Constants for approximating costs of different action types
    MOVE_COST = 1
    MOUNT_COST = 1
    CAPTURE_COST = 1
    UPLOAD_COST = 1
    RECHARGE_COST = 1

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Parse static movement graph between waypoints (kxvuge0a ...)
        self.waypoint_graph = {}
        for fact in self.static:
            if match(fact, "kxvuge0a", "*", "*"):
                _, w1, w2 = get_parts(fact)
                self.waypoint_graph.setdefault(w1, set()).add(w2)

        # Parse mapping from objectives to waypoints (wecz86om)
        self.objective_to_waypoint = {}
        for fact in self.static:
            if match(fact, "wecz86om", "*", "*"):
                _, objective, waypoint = get_parts(fact)
                self.objective_to_waypoint[objective] = waypoint

        # Extract rovers from static (jy67mgmv predicate)
        self.rovers = set()
        for fact in self.static:
            if match(fact, "jy67mgmv", "*"):
                _, rover = get_parts(fact)
                self.rovers.add(rover)

        # Extract stores (v4e7u9dr type) associated to each rover (fa5tam1q ?store ?rover)
        self.store_of_rover = {}
        for fact in self.static:
            if match(fact, "fa5tam1q", "*", "*"):
                _, store, rover = get_parts(fact)
                self.store_of_rover[rover] = store

        # Extract cameras (f3vbbxzk)
        self.cameras = set()
        for fact in self.static:
            if match(fact, "g52pa0bi", "*", "*"):
                # camera with objective (we also use camera names later)
                _, camera, _ = get_parts(fact)
                self.cameras.add(camera)

        # Extract objectives with their required image types from goals:
        # goals of form (lk059nof objective image-type)
        self.objective_goals = {}
        for goal in self.goals:
            if match(goal, "lk059nof", "*", "*"):
                _, obj, image_type = get_parts(goal)
                self.objective_goals.setdefault(obj, set()).add(image_type)

    def _shortest_path_length(self, graph, start, goal):
        """Compute shortest path length between start and goal in graph (unweighted)."""
        if start == goal:
            return 0
        visited = {start}
        frontier = [(start, 0)]
        while frontier:
            current, dist = frontier.pop(0)
            for neighbor in graph.get(current, ()):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, dist + 1))
        # If no path found, return a large cost as penalty but not infinite.
        return 1000

    def __call__(self, node):
        state = node.state

        # Shortcut: if goals are already reached, heuristic is zero
        if self.goals <= state:
            return 0

        # Extract rover current positions: (ilkg22gp ?rover ?waypoint)
        rover_positions = {}
        for fact in state:
            if match(fact, "ilkg22gp", "*", "*"):
                _, rover, waypoint = get_parts(fact)
                rover_positions[rover] = waypoint

        # Extract which stores have pmcrx641 (charged) or zdfck87b (discharged)
        store_charged = set()
        store_discharged = set()
        for fact in state:
            if match(fact, "pmcrx641", "*"):
                _, store = get_parts(fact)
                store_charged.add(store)
            elif match(fact, "zdfck87b", "*"):
                _, store = get_parts(fact)
                store_discharged.add(store)

        # Extract mounted cameras and what cameras are mounted on rovers (v25mqvwj camera rover)
        mounted_camera_on_rover = {}
        for fact in state:
            if match(fact, "v25mqvwj", "*", "*"):
                _, camera, rover = get_parts(fact)
                mounted_camera_on_rover[rover] = camera

        # Extract which cameras exist and their possible image modes (xq9py0w8 camera mode)
        camera_modes = {}
        for fact in self.static:
            if match(fact, "xq9py0w8", "*", "*"):
                _, camera, mode = get_parts(fact)
                camera_modes.setdefault(camera, set()).add(mode)

        # Extract which rovers are jy67mgmv (active)
        active_rovers = set()
        for fact in state:
            if match(fact, "jy67mgmv", "*"):
                _, rover = get_parts(fact)
                active_rovers.add(rover)

        # Extract which waypoints have chfbweyv (visited / image taken on waypoint)
        visited_waypoints = set()
        for fact in state:
            if match(fact, "chfbweyv", "*"):
                _, wp = get_parts(fact)
                visited_waypoints.add(wp)

        heuristic_cost = 0

        # Helper: estimate cost for a rover to take an image of given mode at objective
        def estimate_capture_cost(rover, objective, image_type):
            total = 0

            # 1. Movement cost: rover moves from current pos to objective waypoint
            start_wp = rover_positions.get(rover, None)
            goal_wp = self.objective_to_waypoint.get(objective, None)
            if start_wp is None or goal_wp is None:
                return 1000  # large penalty if missing info
            dist = self._shortest_path_length(self.waypoint_graph, start_wp, goal_wp)
            total += dist * self.MOVE_COST

            # 2. Check if camera mounted on rover and in required mode
            camera = mounted_camera_on_rover.get(rover, None)
            if camera is None or image_type not in camera_modes.get(camera, set()):
                # assume camera mount or change mode needed
                total += self.MOUNT_COST

            # 3. Capture cost (always 1 capture action)
            total += self.CAPTURE_COST

            # 4. Check if store charged, otherwise add recharge cost
            store = self.store_of_rover.get(rover, None)
            if store is None:
                # no store, add penalty
                total += self.RECHARGE_COST
            elif store not in store_charged:
                total += self.RECHARGE_COST

            # 5. Upload cost after capture to store (assume always needed)
            total += self.UPLOAD_COST

            return total

        # For each objective and missing image type, find rover with minimal cost to capture
        for objective, required_image_types in self.objective_goals.items():
            # Determine which image types are already done (lk059nof ...)
            done_image_types = {
                get_parts(fact)[2]
                for fact in state
                if match(fact, "lk059nof", objective, "*")
            }

            missing_images = required_image_types - done_image_types
            if not missing_images:
                continue  # objective fully done

            # For each missing image type, find minimal rover cost
            for image_type in missing_images:
                min_cost_for_image = 10000
                for rover in active_rovers:
                    cost = estimate_capture_cost(rover, objective, image_type)
                    if cost < min_cost_for_image:
                        min_cost_for_image = cost
                heuristic_cost += min_cost_for_image

        # Return the accumulated heuristic cost (finite and >0 unless goal reached)
        return heuristic_cost
