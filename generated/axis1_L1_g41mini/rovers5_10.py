from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract elements from a PDDL fact string by stripping parentheses and splitting by spaces."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern, where pattern elements may contain wildcards '*'.

    Args:
        fact (str): PDDL fact string, e.g. "(predicate arg1 arg2)".
        args (list[str]): Pattern elements to match, e.g. "ilkg22gp", "*", "*".

    Returns:
        bool: True if all parts match the pattern; otherwise False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain (cdmmbhnd).

    Summary:
        This heuristic estimates the number of actions required to achieve all goal facts,
        focusing on the main evidences of progress in the domain:
        - Acquiring (lk059nof objective colour), (lk059nof objective high_res), and (lk059nof objective low_res)
        - Capturing (chfbweyv waypoint) goals.
        It uses the state information about rovers' locations, camera assignments, image data collected,
        and whether waypoints have been "checked" via some action (marked by chfbweyv).

    Assumptions:
        - The main goal facts indicate that images of various qualities (colour, high_res, low_res) of objectives must be obtained.
        - Objectives are represented as zi8ouvk1 objects; cameras take images with specific resolutions/qualities (kad6swsx).
        - Rovers move between waypoints (gwg9gujk), and cameras can be assigned to rovers.
        - Image capturing involves a sequence of actions: rover positioning, camera targeting objective, data acquisition.
        - Checking a waypoint (chfbweyv) is another goal subtask.
        - The heuristic does not need to be admissible, but should be informative and efficiently computable.

    Heuristic Initialization:
        - Extract from static facts the connectivity between waypoints (kxvuge0a) as a graph to estimate shortest distances.
        - Extract camera-objective compatibilities (g52pa0bi) to know which camera can observe which objective.
        - Extract which cameras can be assigned to which rovers (uss6vmx6).
        - Extract which objectives are needed with which image types from goal facts (lk059nof).
        - Extract waypoints needing "checking" from goal facts (chfbweyv).
        - Extract rover movement adjacencies (xuwncpn8).
        - Store rover ownership of stores (fa5tam1q) and status of stores (pmcrx641, zdfck87b).
        - Precompute waypoint adjacency and shortest path distances via BFS.

    Step-By-Step Thinking for Computing Heuristic:
        1. If the state is a goal state (all goal facts hold), return 0.

        2. Compute for each goal objective-image-type triple whether it's achieved by checking lk059nof facts.

        3. For each missing objective-image-type goal:
           a) Find rovers that have cameras capable of that image type.
           b) For such rover-camera pairs, estimate the cost to get that rover to a waypoint from which it can capture images of the objective.
           c) Include costs for any required store readiness (zdfck87b / pmcrx641) and image capturing actions.

        4. Compute for each goal waypoint needing checking (chfbweyv) whether it is achieved; if not,
           estimate minimal rover moves to that waypoint and an action to check it.

        5. The heuristic value is the sum of:
           - Minimal costs to fulfill all missing images goals.
           - Minimal costs to fulfill all remaining waypoint checks.

        6. For movement cost estimates between waypoints, use precomputed shortest path distances in the waypoint graph.

        7. We assume one action per move between adjacent waypoints (based on xuwncpn8), plus constant actions for capturing images or checking waypoints.

    This heuristic should guide greedy best-first search effectively by focusing on nearest required tasks with reasonable cost approximations.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract all waypoints (gwg9gujk) by scanning static facts with predicates
        self.waypoints = set()
        for fact in self.static:
            if match(fact, "kxvuge0a", "*", "*"):
                _, w1, w2 = get_parts(fact)
                self.waypoints.add(w1)
                self.waypoints.add(w2)
            elif match(fact, "chfbweyv", "*"):
                _, wp = get_parts(fact)
                self.waypoints.add(wp)

        # Build an undirected graph of waypoint adjacencies from static "kxvuge0a"
        self.waypoint_neighbors = {wp: set() for wp in self.waypoints}
        for fact in self.static:
            if match(fact, "kxvuge0a", "*", "*"):
                _, w1, w2 = get_parts(fact)
                self.waypoint_neighbors[w1].add(w2)
                self.waypoint_neighbors[w2].add(w1)

        # Precompute shortest path distances between waypoints using BFS
        self.waypoint_distances = {}
        for wp in self.waypoints:
            self.waypoint_distances[wp] = self._bfs_distances(wp)

        # Extract cameras (f3vbbxzk), objectives (zi8ouvk1), and image types (kad6swsx)
        # Information from static to associate:
        # g52pa0bi camera objective : camera can observe objective
        self.camera_to_objectives = {}
        self.objectives = set()
        self.cameras = set()
        for fact in self.static:
            if match(fact, "g52pa0bi", "*", "*"):
                _, camera, obj = get_parts(fact)
                self.camera_to_objectives.setdefault(camera, set()).add(obj)
                self.objectives.add(obj)
                self.cameras.add(camera)

        # From static: uss6vmx6 camera rover : camera assigned to rover
        self.camera_to_rovers = {}
        self.rovers = set()
        for fact in self.static:
            if match(fact, "uss6vmx6", "*", "*"):
                _, camera, rover = get_parts(fact)
                self.camera_to_rovers.setdefault(camera, set()).add(rover)
                self.rovers.add(rover)

        # From static: fa5tam1q store rover : store belongs to rover
        self.store_of_rover = {}
        for fact in self.static:
            if match(fact, "fa5tam1q", "*", "*"):
                _, store, rover = get_parts(fact)
                self.store_of_rover[rover] = store

        # From static: pmcrx641 store (ready) and zdfck87b store (not ready?)
        # Track stores and their readiness status by facts in state during heuristic calculation
        # We'll handle these during runtime.

        # From static: xuwncpn8 rover from_wpt to_wpt (rover movement between waypoints)
        # Build movement graph per rover (mostly uniform for all rovers, assume edges same)
        self.movement_graph = {wp: set() for wp in self.waypoints}
        for fact in self.static:
            if match(fact, "xuwncpn8", "*", "*", "*"):
                # rover, from_wpt, to_wpt
                _, _, from_wpt, to_wpt = get_parts(fact)
                self.movement_graph[from_wpt].add(to_wpt)

        # From static: e9bo95dc general waypoint (general location of the general)
        # Not necessarily needed here for heuristics, ignore for now.

        # Image qualities: kad6swsx types appear in goal facts, e.g. low_res, high_res, colour
        self.image_types = set()
        for fact in self.goals:
            if match(fact, "lk059nof", "*", "*"):
                _, obj, img_type = get_parts(fact)
                self.image_types.add(img_type)
                self.objectives.add(obj)

        # Track which waypoints require checking (chfbweyv) goals:
        self.waypoints_to_check = set()
        for fact in self.goals:
            if match(fact, "chfbweyv", "*"):
                _, wp = get_parts(fact)
                self.waypoints_to_check.add(wp)

    def _bfs_distances(self, start):
        """Return dict of minimal moves from start waypoint to every other waypoint using waypoint adjacency."""
        from collections import deque

        distances = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.waypoint_neighbors.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        state = node.state

        # Goal test: if all goal facts hold in state, heuristic = 0
        if self.goals <= state:
            return 0

        # Helper to extract rovers' current waypoint by ilkg22gp facts:
        # (ilkg22gp rover waypoint)
        rover_positions = {}
        for fact in state:
            if match(fact, "ilkg22gp", "*", "*"):
                _, rover, wpt = get_parts(fact)
                rover_positions[rover] = wpt

        # Extract which stores are ready: pmcrx641 (ready) and zdfck87b (not ready)
        store_ready = set()
        for fact in state:
            if match(fact, "pmcrx641", "*"):
                _, store = get_parts(fact)
                store_ready.add(store)

        # Extract which cameras are currently assigned to rovers (uss6vmx6)
        # Use static info - but this doesn't change, we rely on static associations

        # Extract image facts:
        # lk059nof objective image_type means objective has image of that type
        images_acquired = set()
        for fact in state:
            if match(fact, "lk059nof", "*", "*"):
                _, obj, img_type = get_parts(fact)
                images_acquired.add((obj, img_type))

        # Extract waypoints checked: chfbweyv waypoint
        waypoints_checked = set()
        for fact in state:
            if match(fact, "chfbweyv", "*"):
                _, wp = get_parts(fact)
                waypoints_checked.add(wp)

        total_cost = 0

        # === Estimate cost for missing image goals (lk059nof) ===

        # Collect missing image goals
        missing_images = []
        for fact in self.goals:
            if match(fact, "lk059nof", "*", "*"):
                _, obj, img_type = get_parts(fact)
                if (obj, img_type) not in images_acquired:
                    missing_images.append((obj, img_type))

        # For each missing image, estimate minimal cost to fulfill it
        for (obj, img_type) in missing_images:

            # Identify cameras that can image that objective and support that image_type
            candidate_cameras = []
            for cam in self.cameras:
                if obj in self.camera_to_objectives.get(cam, ()):
                    # Check if camera supports the image_type (xq9py0w8 camera image_type)
                    # This is a static fact: xq9py0w8 camera img_type
                    found_img_type = False
                    for fact in self.static:
                        if match(fact, "xq9py0w8", cam, img_type):
                            found_img_type = True
                            break
                    if found_img_type:
                        candidate_cameras.append(cam)

            if not candidate_cameras:
                # No camera can take required image, this should never happen in a solvable instance,
                # but to remain finite return a large cost:
                total_cost += 30
                continue

            # For each candidate camera, find rovers it can be assigned to,
            # and rover current waypoint:
            best_cost_for_image = float("inf")
            for cam in candidate_cameras:
                rovers_for_cam = self.camera_to_rovers.get(cam, set())
                for rover in rovers_for_cam:
                    if rover not in rover_positions:
                        continue  # Rover position unknown, skip

                    rover_wp = rover_positions[rover]

                    # We want to find a waypoint from which rover can image 'obj'
                    # based on wecz86om objective waypoint (objective at waypoint)
                    # Get all waypoints of objective from static or static + state?
                    # The wecz86om predicate appears in static and can appear in facts.
                    # We'll collect all waypoints of the objective from both static and state.
                    obj_waypoints = set()
                    for fact in self.static | state:
                        if match(fact, "wecz86om", obj, "*"):
                            _, _, wpt = get_parts(fact)
                            obj_waypoints.add(wpt)

                    if not obj_waypoints:
                        # Objective has no waypoints? To avoid infinite, assign large cost.
                        cost = 20
                        if cost < best_cost_for_image:
                            best_cost_for_image = cost
                        continue

                    # Calculate minimal rover moves from rover_wp to any objective waypoint
                    min_move_dist = float("inf")
                    for wpt in obj_waypoints:
                        dist = self.waypoint_distances.get(rover_wp, {}).get(wpt, float("inf"))
                        if dist < min_move_dist:
                            min_move_dist = dist

                    if min_move_dist == float("inf"):
                        # No path found, large penalty
                        min_move_dist = 20

                    # Check if store of rover is ready (pmcrx641), else cost to prepare it (assume 2 actions)
                    store = self.store_of_rover.get(rover, None)
                    store_cost = 0 if (store and store in store_ready) else 2

                    # Assume capturing image involves a fixed cost of 2 actions (to represent actual capture)
                    capture_cost = 2

                    # Total cost for this camera-rover combo:
                    cost = min_move_dist + store_cost + capture_cost

                    if cost < best_cost_for_image:
                        best_cost_for_image = cost

            # Add minimal cost for this image goal
            total_cost += best_cost_for_image

        # === Estimate cost for missing waypoint checks (chfbweyv) ===

        unchecked_wps = self.waypoints_to_check - waypoints_checked

        # For each unchecked waypoint, cost is minimal move of some rover + 1 for checking action.

        # If no rovers or no positions, assign a large penalty:
        if not rover_positions:
            total_cost += 10 * len(unchecked_wps)
        else:
            rover_positions_list = list(rover_positions.values())
            for wp in unchecked_wps:
                min_dist = float("inf")
                for rover_wp in rover_positions_list:
                    dist = self.waypoint_distances.get(rover_wp, {}).get(wp, float("inf"))
                    if dist < min_dist:
                        min_dist = dist

                if min_dist == float("inf"):
                    min_dist = 10  # large penalty if unreachable

                total_cost += min_dist + 1  # add 1 action for checking waypoint

        # Return finite heuristic estimate
        return total_cost
