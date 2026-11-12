from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the parts of a fact by removing surrounding parentheses and splitting by whitespace.
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a fact matches a given pattern with possible wildcards "*" in args.
    Returns True if all parts match.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (cdmmbhnd).

    Summary:
    This heuristic estimates the number of actions required to satisfy all goals
    for the secret_domain by focusing on the achievement of "lk059nof" predicates
    for objectives with specific image types (e.g. high_res, low_res, colour)
    and the associated necessary data collection and rover positioning steps.
    It approximates the cost by counting:
    - The number of unfinished objectives that require image acquisition in the specified resolutions/colors.
    - The rover's need to be at the appropriate waypoints to capture images.
    - The possibility of needing to collect or process images in rover stores.
    The heuristic ignores complex interleavings but provides a meaningful informed estimate to guide search efficiently.

    Assumptions:
    - Goals are mainly conjunctions of facts of the form (lk059nof objective image_type).
    - Taking images requires the rover to be at a waypoint, have the camera ready, and the objective linked accordingly.
    - The rover must move between waypoints to capture images.
    - Cameras can only handle certain image types linked to objectives.
    - Relevant static connectivity and relationships (e.g. waypoint links, camera-objective associations) are considered.
    - Heuristic need not be admissible but should be efficient and reduce search node expansions.

    Heuristic Initialization:
    - Extract all goal predicates of type "lk059nof" with their objectives and image types.
    - Extract static facts:
      * Waypoint connectivity: kxvuge0a links between waypoints.
      * Rover's allowed moves between waypoints: xuwncpn8 facts.
      * Objectives' association with waypoints: wecz86om facts.
      * Cameras' capabilities of capturing objectives: g52pa0bi.
      * Camera-resolutions: xq9py0w8.
      * Rovers and store relations: fa5tam1q.
    - Build fast lookup maps for:
      * Waypoint adjacency graph for rover moves.
      * Objectivs' reachable waypoints.
      * Cameras that can take images for each objective-image_type pair.
      * Current rover location and store status (to be used during heuristic call).
    - Use shortest path approximations via breadth-first search for moving the rover between waypoints.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify which goals are not yet achieved in the current state.
    2. For each required (objective, image_type) goal not achieved:
       a. Identify waypoints where the rover can take that image (via cameras able to capture that objective and image_type).
       b. Determine the rover's current waypoint.
       c. Estimate the shortest move cost for the rover to reach a suitable waypoint.
       d. Add a fixed cost for "taking the image" action(s), assumed constant per image.
    3. Sum over all such needed images.
    4. Additionally, consider if the rover's store needs to be refreshed/prepared (e.g. pmcrx641 predicate) by checking state facts and accounting for associated actions (approximated as a small fixed cost per store reset).
    5. Return the total estimated cost. For goal states, this is zero.

    The heuristic balances computational efficiency by using cached static info and approximate shortest path costs, avoiding full relaxed planning graph computations.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract all goal objectives and their image types from (lk059nof objective image_type)
        self.goal_images = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "lk059nof" and len(parts) == 3:
                _, objective, img_type = parts
                self.goal_images.add((objective, img_type))

        # Build adjacency graph of waypoints using static facts kxvuge0a (links)
        self.waypoint_graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "kxvuge0a" and len(parts) == 3:
                w1, w2 = parts[1], parts[2]
                self.waypoint_graph.setdefault(w1, set()).add(w2)
                # Note: Links may or may not be symmetric, keep as is.

        # Note which waypoints the rover can move between (using xuwncpn8 facts)
        # key: rover -> {(from_wp, to_wp)}
        self.rover_moves = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "xuwncpn8" and len(parts) == 4:
                rover, wp_from, wp_to = parts[1], parts[2], parts[3]
                self.rover_moves.setdefault(rover, set()).add((wp_from, wp_to))

        # Map each objective to waypoints reachable with (wecz86om objective waypoint)
        self.objective_waypoints = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "wecz86om" and len(parts) == 3:
                obj, wp = parts[1], parts[2]
                self.objective_waypoints.setdefault(obj, set()).add(wp)

        # Which cameras can capture which objectives (g52pa0bi camera objective)
        self.camera_objectives = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "g52pa0bi" and len(parts) == 3:
                cam, obj = parts[1], parts[2]
                self.camera_objectives.setdefault(cam, set()).add(obj)

        # Which image types cameras support (xq9py0w8 camera image_type)
        self.camera_img_types = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "xq9py0w8" and len(parts) == 3:
                cam, img_type = parts[1], parts[2]
                self.camera_img_types.setdefault(cam, set()).add(img_type)

        # Rovers and their associated stores (fa5tam1q store rover)
        self.store_to_rover = {}
        self.rover_to_store = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "fa5tam1q" and len(parts) == 3:
                store, rover = parts[1], parts[2]
                self.store_to_rover[store] = rover
                self.rover_to_store[rover] = store

        # Build: For each (objective, image_type), which waypoints have a camera that can take it?
        # This is computed on demand in __call__ to limit memory for big domains.

    def shortest_path(self, rover, start, goals):
        """
        Compute shortest path length from start waypoint to any waypoint in goals
        using rover allowed moves (xuwncpn8).
        Returns integer distance or large number if unreachable.
        """
        if start in goals:
            return 0
        moves = self.rover_moves.get(rover, set())
        from collections import deque

        visited = set([start])
        queue = deque([(start, 0)])

        while queue:
            current, dist = queue.popleft()
            for (frm, to) in moves:
                if frm == current and to not in visited:
                    if to in goals:
                        return dist + 1
                    visited.add(to)
                    queue.append((to, dist + 1))
        return 10000  # unreachable large cost

    def cameras_for_objective_image(self, objective, image_type):
        """
        Return set of cameras that can capture objective in image_type.
        """
        cams = set()
        for cam, objs in self.camera_objectives.items():
            if objective in objs and image_type in self.camera_img_types.get(cam, set()):
                cams.add(cam)
        return cams

    def waypoints_for_camera_rover(self, rover, cam):
        """
        Return waypoints where rover can operate that contain the given camera.
        Approximate by assuming rover may only take images at current waypoint,
        so we consider that the camera must be on the rover.
        But domain doesn't explicitly link camera location,
        thus we use a heuristic: the rover can use any camera it "has",
        and this is approximated as the rover itself.
        Our heuristic assumes camera is always with the rover.
        """
        # The domain doesn't specify where cameras are located,
        # only which rover they belong to (uss6vmx6).
        # We'll approximate that rover can take images from any waypoint it can be.
        return set(self.waypoint_graph.keys())  # underapproximate: all waypoints

    def waypoints_for_objective_image(self, rover, objective, image_type):
        """
        Collect possible waypoints where the rover can take the image of (objective, image_type).
        These are waypoints where:
        - The objective is at or accessible (wecz86om)
        - The rover can move there (rover_moves)
        - The rover has a camera that can take the required image.
        """
        cams = self.cameras_for_objective_image(objective, image_type)
        if not cams:
            return set()  # no camera can do that image: no solution from here.

        # Waypoints associated with the objective (where objective is present)
        obj_wps = self.objective_waypoints.get(objective, set())

        # Consider only waypoints reachable by rover (keys in rover_moves)
        # We simplify and say rover can be in any waypoint in graph reachable somehow.

        # Combine sets: waypoints where objective is, intersect with all waypoints
        # (since cameras assumed with rover, rover can be anywhere)
        return obj_wps

    def __call__(self, node):
        """
        Estimate required number of actions to reach goal:
        Steps:
        - If state is goal, heuristic = 0.
        - For each goal (lk059nof obj img_type) not in state:
          * find candidate waypoints to take picture.
          * estimate shortest move cost from current rover position.
          * add fixed cost for taking the image.
        - Consider cost for preparing camera stores if needed.
        """
        state = node.state

        if self.goals <= state:
            return 0

        # Find rover(s) in state:
        # State facts (ilkg22gp rover waypoint) show rover positions.
        rovers = set()
        rover_positions = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ilkg22gp" and len(parts) == 3:
                rover, wp = parts[1], parts[2]
                rovers.add(rover)
                rover_positions[rover] = wp

        if not rovers:
            # no rover position found, heuristic unknown; return large cost
            return 10000

        total_cost = 0

        # We'll sum costs over all rovers and all missing images they must capture.
        # Since the example appears to have a single rover typically, we treat each rover separately.

        # Fixed costs assumptions for actions:
        move_cost = 1    # approx. cost of moving between waypoints
        take_image_cost = 2  # approximate cost for taking an image (capturing, processing)
        prepare_store_cost = 3  # cost for actions related to preparing rover's store (like cp8otkvs or o64yr71v)

        # For each rover, estimate cost to fulfill all missing images.
        for rover in rovers:
            rover_wp = rover_positions.get(rover)
            if rover_wp is None:
                # unknown rover position - pessimistic estimate
                return 10000

            # Get store for this rover
            store = self.rover_to_store.get(rover)

            # Check if store is ready (pmcrx641 store) or if needs preparing (zdfck87b store not in state)
            # The store is ready if (pmcrx641 store) in state
            store_ready = False
            if store and ("(pmcrx641 {})".format(store) in state):
                store_ready = True

            # If store not ready, add its cost once
            if store and not store_ready:
                total_cost += prepare_store_cost

            # For each goal image not yet satisfied
            for (obj, img_type) in self.goal_images:
                goal_fact = f"(lk059nof {obj} {img_type})"
                if goal_fact not in state:
                    # Find candidate waypoints where rover can take this image
                    candidate_wps = self.waypoints_for_objective_image(rover, obj, img_type)
                    if not candidate_wps:
                        # No place to take this image: large cost
                        total_cost += 10000
                        continue

                    # Compute shortest path for rover from rover_wp to any candidate waypoint
                    dist = self.shortest_path(rover, rover_wp, candidate_wps)
                    if dist > 9000:
                        # Unreachable image location
                        total_cost += 10000
                        continue

                    # Cost estimated as move actions + take image action
                    total_cost += dist * move_cost + take_image_cost

        return total_cost
