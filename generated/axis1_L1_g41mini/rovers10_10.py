from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a fact matches a given pattern with wildcards.

    Args:
        fact (str): PDDL fact as string, e.g. "(predicate obj1 obj2)".
        args: Pattern parts to match with wildcards '*' allowed.

    Returns:
        bool: True if fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (cdmmbhnd).

    Summary
    -------
    Estimates the number of required actions to reach the goal, focusing on satisfying
    goals of form (lk059nof objective colour). The heuristic guides the search by 
    counting unresolved goals and estimating the minimum maneuvering needed for rovers
    to capture required images at waypoints with necessary resolutions.

    Assumptions
    -----------
    - Each goal fact is of the form (lk059nof objective resolution).
    - To achieve (lk059nof objective resolution), rovers must capture images
      (l3943stc rover objective resolution) first, via actions requiring the rover 
      to be at a waypoint with the objective and correct camera data.
    - Rovers need to move between waypoints; moves are possible according to static 
      facts (kxvuge0a) defining connectivity between waypoints.
    - Rovers carry cameras and store data in associated storage units.
    - Cameras must be "v25mqvwj camera rover" and use "l3943stc rover objective resolution" 
      to achieve the imaging goal.
    - Static facts encode connectivity and capabilities used to estimate shortest paths.
    - The heuristic ignores exact preconditions but uses connectivity and missing goals 
      to estimate progress cost.
    - The heuristic is not admissible but is efficiently computable and informative.

    Heuristic Initialization
    ------------------------
    - Extract goals of type (lk059nof objective resolution).
    - Extract static facts:
      * Waypoint connectivity (kxvuge0a) to build a waypoint graph.
      * Facts relating rovers to waypoints and cameras to rovers.
    - Precompute shortest paths between waypoints using connectivity.
    - Extract rovers and their current waypoints from the state.
    - Extract cameras and their capabilities (which resolutions they can achieve).
    - Map objectives to their associated waypoints from static facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal (lk059nof objective resolution):
      1. Check if goal already achieved in current state: then cost=0 for this goal.
      2. Otherwise:
         - Identify all rovers that can capture the required resolution of the objective.
           (via cameras they carry and camera-objective links from static).
         - For each capable rover:
           a. Find current rover position (waypoint).
           b. Find possible waypoints related to the objective from (wecz86om objective waypoint).
           c. Compute minimal travel cost (shortest waypoint path length) from rover position 
              to any such objective waypoint.
           d. Add fixed "capture cost" (1 or 2 steps, e.g. capture action) to represent imaging.
         - The minimal of these costs among rovers is taken as the estimated cost for that goal.
    - Sum all per-goal estimated costs to get the heuristic estimate.
    - This roughly bounds minimal moves plus required captures to complete the goal.

    This heuristic smoothly accounts for:
      - Multiple rovers deployed at different waypoints,
      - Multiple resolutions per objective,
      - Connectivity between waypoints,
      - Cameras carried by rovers and their capabilities,
      - Location of objectives in the map,
    providing informative guidance for greedy best-first search.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract waypoint connectivity graph: edges from kxvuge0a (waypoint1 waypoint2)
        self.waypoint_graph = {}
        for fact in self.static:
            if match(fact, "kxvuge0a", "*", "*"):
                _, wp_from, wp_to = get_parts(fact)
                self.waypoint_graph.setdefault(wp_from, set()).add(wp_to)

        # Extract rovers known in domain: objects typed cdmmbhnd appear in static facts:
        # e.g. (dp3yzgup rover), (vg9nzfll rover), (jy67mgmv rover) are predicates over rovers.
        self.rovers = set()
        for fact in self.static:
            # heuristically identify rovers; e.g. predicates dp3yzgup, vg9nzfll, jy67mgmv use rover arg
            if match(fact, "dp3yzgup", "*"):
                self.rovers.add(get_parts(fact)[1])
            elif match(fact, "vg9nzfll", "*"):
                self.rovers.add(get_parts(fact)[1])
            elif match(fact, "jy67mgmv", "*"):
                self.rovers.add(get_parts(fact)[1])
            elif match(fact, "fa5tam1q", "*", "*"):
                # rover stores associated to rovers
                store, rover = get_parts(fact)[1], get_parts(fact)[2]
                self.rovers.add(rover)

        # Extract cameras and their objectives:
        # camera-objective links via g52pa0bi camera objective
        self.camera_objectives = {}  # camera -> set of objectives
        for fact in self.static:
            if match(fact, "g52pa0bi", "*", "*"):
                _, camera, objective = get_parts(fact)
                self.camera_objectives.setdefault(camera, set()).add(objective)

        # Extract the capabilities (resolutions/colors) cameras can support: xq9py0w8 camera resolution
        self.camera_caps = {}
        for fact in self.static:
            if match(fact, "xq9py0w8", "*", "*"):
                _, camera, res = get_parts(fact)
                self.camera_caps.setdefault(camera, set()).add(res)

        # Map objectives to their waypoints from wecz86om objective waypoint
        self.objective_waypoints = {}
        for fact in self.static:
            if match(fact, "wecz86om", "*", "*"):
                _, objective, waypoint = get_parts(fact)
                self.objective_waypoints.setdefault(objective, set()).add(waypoint)

        # Precompute shortest distances between waypoints using BFS for efficiency.
        self._precompute_shortest_paths()

    def _precompute_shortest_paths(self):
        """Precompute shortest paths between all reachable waypoints using BFS."""

        # Store shortest distances as dict of dict: wp_from -> wp_to -> distance
        self.shortest_distances = {}

        for start in self.waypoint_graph:
            dist = {start: 0}
            queue = [start]
            while queue:
                current = queue.pop(0)
                for neighbor in self.waypoint_graph.get(current, []):
                    if neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)
            self.shortest_distances[start] = dist

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Get rovers' current positions (waypoints) from ilkg22gp rover waypoint
        rover_positions = {}
        for fact in state:
            if match(fact, "ilkg22gp", "*", "*"):
                _, rover, waypoint = get_parts(fact)
                if rover in self.rovers:
                    rover_positions[rover] = waypoint

        # Get cameras associated to rovers from uss6vmx6 camera rover
        rover_cameras = {}  # rover -> set of cameras
        for fact in state:
            if match(fact, "uss6vmx6", "*", "*"):
                _, camera, rover = get_parts(fact)
                rover_cameras.setdefault(rover, set()).add(camera)

        # Determine which (objective, resolution) goals remain unmet
        goals_needed = []
        for goal_fact in self.goals:
            if goal_fact not in state:
                # Only consider (lk059nof objective resolution) goals
                if match(goal_fact, "lk059nof", "*", "*"):
                    _, objective, resolution = get_parts(goal_fact)
                    goals_needed.append((objective, resolution))

        total_cost = 0

        # Fixed heuristic cost representing the capture action
        capture_cost = 1

        # For each needed goal, estimate cost to achieve it:
        for objective, resolution in goals_needed:
            # If objective not in any known objective waypoint, heuristic fallback cost 5
            objective_wps = self.objective_waypoints.get(objective, None)
            if not objective_wps:
                # No knowledge of objective location -> arbitrary penalty
                total_cost += 5
                continue

            # Find minimal cost among capable rovers
            minimal_goal_cost = None

            for rover in self.rovers:
                # Check if rover has position
                rover_wp = rover_positions.get(rover, None)
                if rover_wp is None:
                    # rover location unknown; assign a penalty cost to represent unknown position
                    continue

                # Check cameras rover carries that can achieve this objective with required resolution
                cameras_on_rover = rover_cameras.get(rover, set())
                capable_cameras = [
                    cam for cam in cameras_on_rover
                    if objective in self.camera_objectives.get(cam, set())
                    and resolution in self.camera_caps.get(cam, set())
                ]
                if not capable_cameras:
                    # No capable camera on this rover for this resolution-objective
                    continue

                # Compute shortest travel distance to any waypoint relevant to objective
                # Among all objective waypoints, find minimum distance from rover_wp
                distances_from_rover = self.shortest_distances.get(rover_wp, {})

                min_dist_to_objective = None
                for obj_wp in objective_wps:
                    dist = distances_from_rover.get(obj_wp, None)
                    if dist is not None:
                        if min_dist_to_objective is None or dist < min_dist_to_objective:
                            min_dist_to_objective = dist

                if min_dist_to_objective is None:
                    # No path found to objective waypoint for this rover.
                    # Assign a large penalty (e.g. 10)
                    est_cost = 10 + capture_cost
                else:
                    # Cost = movement steps + capture action
                    est_cost = min_dist_to_objective + capture_cost

                if minimal_goal_cost is None or est_cost < minimal_goal_cost:
                    minimal_goal_cost = est_cost

            if minimal_goal_cost is None:
                # No rover can achieve this goal (no capable rover or unknown positions)
                # Assign a high penalty heuristic to encourage exploration to fix this
                minimal_goal_cost = 15

            total_cost += minimal_goal_cost

        # Return total estimated cost for all missing goals
        return total_cost
