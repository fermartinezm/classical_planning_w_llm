from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    """
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
    A domain-dependent heuristic for the secret_domain domain (cdmmbhnd).

    Summary:
    The heuristic estimates the number of required actions to satisfy
    all goal facts related to imaging objectives with specific image qualities.
    The domain models rovers moving between waypoints, equipped with cameras supporting various
    image qualities, and objectives that must be imaged with different resolutions and color types.
    The heuristic focuses on evaluating missing images for each objective and estimating
    the minimal actions needed, including movement and equipment preparation costs.

    Assumptions:
    - The goals require capturing images of objectives in specific qualities, represented by (lk059nof objective quality).
    - A rover must be present at a waypoint connected to the objective to take an image.
    - Cameras are mounted on rovers and can capture different qualities depending on the configuration.
    - Rovers can move and switch stored equipment (modeled via predicates and actions).
    - Movement is restricted by waypoints connectivity encoded by (kxvuge0a wp1 wp2) and rover movement predicates (xuwncpn8).
    - Each image is taken by a specific rover/camera-objective pair (evident from effects).
    - Static facts encode connectivity between waypoints and available imaging setups.
    - The heuristic does not require admissibility, but aims to minimize expanded nodes by approximating:
      - The minimal moves for rovers from their current position to appropriate locations.
      - The minimal preparation steps to image outstanding goals.

    Heuristic Initialization:
    - Extract sets of waypoints and their connectivity graph.
    - Extract objectives and their required goal qualities.
    - Extract rovers and their current known locations and mobility structure.
    - Extract camera-objective capabilities.
    - Extract static facts that allow movement transitions and camera imaging properties.
    - Precompute shortest distances between waypoints for efficient rover travel cost estimation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse goals, which are facts of form (lk059nof objective quality) indicating needed images.
    2. Identify which goals are already achieved to only count pending ones.
    3. For each pending image goal:
       - Determine the rovers able to capture that image (using camera capability and rover mounting).
       - For each such rover:
         - Compute the minimal movement cost for that rover from its current waypoint to a waypoint
           suitable to image the objective (according to predicates like wecz86om objective waypoint)
           that matches the goal requirements.
       - Estimate the minimal cost among rovers for that goal.
       - Add minimal estimated action costs for required preparation (e.g. capturing the image, changing equipment).
    4. Sum these minimal costs over all pending image goals.
    5. Return zero if all goals are achieved.

    Notes on cost estimation:
    - Moving one waypoint along the route costs 1 action.
    - Capturing an image is assumed to cost at least 1 action.
    - Equipment changes or camera mode activations could cost extra actions; a small overhead is added.
    - Rover capability and camera configuration constraints filter feasible rover-objective pairs.
    - The heuristic uses shortest paths computed from static waypoint connectivity.

    This heuristic balances computational efficiency (precomputed shortest paths, set lookups) and accuracy
    by considering individual pending goals separately and summing their minimal achievable costs.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static_facts = task.static
        self.operators = task.operators

        # Parse all waypoints (gwg9gujk) from static facts involving kxvuge0a (connectivity)
        self.waypoints = set()
        for fact in self.static_facts:
            if match(fact, "kxvuge0a", "*", "*"):
                parts = get_parts(fact)
                self.waypoints.add(parts[1])
                self.waypoints.add(parts[2])

        # Build undirected graph of waypoint connectivity for shortest path calculation
        self.adj = {wp: set() for wp in self.waypoints}
        for fact in self.static_facts:
            if match(fact, "kxvuge0a", "*", "*"):
                parts = get_parts(fact)
                self.adj[parts[1]].add(parts[2])
                self.adj[parts[2]].add(parts[1])

        # Extract objectives and qualities from goals of the form (lk059nof objective quality)
        self.goal_images = []
        for goal in self.goals:
            if match(goal, "lk059nof", "*", "*"):
                parts = get_parts(goal)
                self.goal_images.append((parts[1], parts[2]))

        # Extract camera-objective capability: (g52pa0bi camera objective)
        self.camera_objective_map = {}
        for fact in self.static_facts:
            if match(fact, "g52pa0bi", "*", "*"):
                parts = get_parts(fact)
                camera, objective = parts[1], parts[2]
                if objective not in self.camera_objective_map:
                    self.camera_objective_map[objective] = set()
                self.camera_objective_map[objective].add(camera)

        # Map cameras to rovers (uss6vmx6 camera rover)
        self.camera_rover_map = {}
        for fact in self.static_facts:
            if match(fact, "uss6vmx6", "*", "*"):
                parts = get_parts(fact)
                camera, rover = parts[1], parts[2]
                if rover not in self.camera_rover_map:
                    self.camera_rover_map[rover] = set()
                self.camera_rover_map[rover].add(camera)

        # Extract waypoints linked to objectives: (wecz86om objective waypoint)
        self.objective_waypoints = {}
        for fact in self.static_facts:
            if match(fact, "wecz86om", "*", "*"):
                parts = get_parts(fact)
                objective, wp = parts[1], parts[2]
                if objective not in self.objective_waypoints:
                    self.objective_waypoints[objective] = set()
                self.objective_waypoints[objective].add(wp)

        # Precompute shortest paths using BFS for all waypoints for efficiency
        self.shortest_paths = self._precompute_all_pairs_shortest_paths(self.waypoints, self.adj)

    def _precompute_all_pairs_shortest_paths(self, nodes, adjacency):
        """
        Precompute shortest distances between all pairs of nodes using BFS.

        Returns a dict of dict: distances[src][dst] = length of shortest path.
        """
        distances = {}
        for src in nodes:
            dist = {src: 0}
            queue = [src]
            while queue:
                current = queue.pop(0)
                for neighbor in adjacency[current]:
                    if neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)
            distances[src] = dist
        return distances

    def __call__(self, node):
        state = node.state

        # If all goals already satisfied, heuristic is zero
        if self.goals <= state:
            return 0

        # Helper to extract rovers currently at which waypoint: (ilkg22gp rover waypoint)
        rover_locations = {}
        for fact in state:
            # Detect rover positions: (ilkg22gp ?rover ?waypoint)
            if match(fact, "ilkg22gp", "*", "*"):
                parts = get_parts(fact)
                rover, wp = parts[1], parts[2]
                rover_locations[rover] = wp

        # Detect mounted cameras via (v25mqvwj camera rover) in current state
        rover_cameras = {}
        for fact in state:
            if match(fact, "v25mqvwj", "*", "*"):
                parts = get_parts(fact)
                camera, rover = parts[1], parts[2]
                if rover not in rover_cameras:
                    rover_cameras[rover] = set()
                rover_cameras[rover].add(camera)

        # Because a camera may be mounted or not, if no mounted camera info, fallback to static mapping
        for rover in rover_locations:
            if rover not in rover_cameras:
                # Use static mapping if available
                cams = self.camera_rover_map.get(rover, set())
                if cams:
                    rover_cameras[rover] = cams
                else:
                    rover_cameras[rover] = set()

        total_estimated_cost = 0

        # For each goal image, estimate minimal cost to achieve it
        for objective, quality in self.goal_images:
            goal_fact = f"(lk059nof {objective} {quality})"
            if goal_fact in state:
                # Already achieved; skip
                continue

            # The objective waypoint candidates where the rover can image the objective
            obj_wps = self.objective_waypoints.get(objective, set())
            if not obj_wps:
                # No known waypoint for the objective — penalize heavily (domain incomplete?)
                total_estimated_cost += 50
                continue

            # Cameras able to image this objective (static)
            cameras_for_obj = self.camera_objective_map.get(objective, set())
            if not cameras_for_obj:
                # No capable cameras — penalize heavily
                total_estimated_cost += 50
                continue

            # Find minimal cost among rovers who can image this objective with the right quality
            minimal_cost_for_goal = None

            for rover, rover_wp in rover_locations.items():
                cameras_on_rover = rover_cameras.get(rover, set())
                # Intersection to find if rover has any camera for the objective
                if not cameras_for_obj.intersection(cameras_on_rover):
                    continue  # this rover can't capture this objective

                # For each waypoint connected to the objective, compute shortest path cost from rover wp
                reachable_wp_costs = []
                sp_from_rover = self.shortest_paths.get(rover_wp, {})
                for obj_wp in obj_wps:
                    dist = sp_from_rover.get(obj_wp)
                    if dist is not None:
                        reachable_wp_costs.append(dist)

                if not reachable_wp_costs:
                    # Not reachable from rover position — assume large cost
                    cost_to_reach = 50
                else:
                    cost_to_reach = min(reachable_wp_costs)

                # Estimated costs:
                # - Movement cost = cost_to_reach
                # - Capturing image cost = 1 action (minimal needed)
                # - Equipment setup or mode switching cost = 1 action assumed overhead
                est_cost = cost_to_reach + 1 + 1

                if minimal_cost_for_goal is None or est_cost < minimal_cost_for_goal:
                    minimal_cost_for_goal = est_cost

            if minimal_cost_for_goal is None:
                # No rover currently able to image the objective => large penalty 
                minimal_cost_for_goal = 50

            total_estimated_cost += minimal_cost_for_goal

        return total_estimated_cost
