from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by stripping parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards.

    :param fact: fact string, e.g. "(ilkg22gp rover1 waypoint2)"
    :param args: pattern parts e.g. "ilkg22gp", "*", "waypoint2"
    :return: True if fact matches pattern, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (cdmmbhnd) domain.

    Summary:
    This heuristic estimates the number of actions required to achieve all goal
    facts related to imaging objectives with specific resolutions (e.g., high_res,
    low_res, colour) for objectives in the secret_domain. The domain models rovers
    moving between waypoints and taking images of objectives in various resolutions
    using cameras. Imaging actions depend on rover locations, camera assignments,
    and data transfer/storage availability.

    Assumptions:
    - Goals involve facts of the form (lk059nof ?objective ?resolution), meaning
      that the objective is imaged at that resolution.
    - Cameras can take images of objectives from specific waypoints if the rover
      is at the corresponding waypoint and has required data stored.
    - Rovers move between waypoints along defined edges (kxvuge0a).
    - Data transfers between rovers and stores require available storage space.
    - Operations related to imaging require the rover to have the right equipment
      and state (captured by predicates in the state).
    - The heuristic is not guaranteed to be admissible but aims to guide the search
      efficiently by estimating the number of incomplete goal subgoals and the
      approximate cost to achieve them.

    Heuristic Initialization:
    - Extract all goal facts of type (lk059nof ?objective ?resolution).
    - Extract static connectivity between waypoints using the (kxvuge0a ?from ?to)
      predicate to build a graph of waypoint adjacency.
    - Extract which rovers are available and their stores.
    - Extract cameras and their characteristics (e.g., which objectives can be imaged).
    - Precompute a waypoint adjacency map for fast shortest-path computations.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify which imaging goals have not been completed yet (missing from current state).
    2. For each missing imaging goal (lk059nof objective res):
       a. Determine rovers that can potentially achieve this imaging goal (have the required camera and objective compatibility).
       b. For each candidate rover:
          i. Estimate the shortest distance in waypoints from the rover's current location to a waypoint from which the rover can image the objective at the required resolution.
          ii. Estimate cost for the rover to acquire and store data (including storage readiness if needed).
          iii. Estimate cost for imaging action itself.
       c. Take the minimal estimated cost over all candidate rovers.
    3. Sum these minimal estimated costs for all missing imaging goals.
    4. Add a penalty for actions to prepare the rover's data store (e.g., toggling ready flags).
    5. Return the total as the heuristic estimate.

    The heuristic uses static connectivity to efficiently compute shortest paths between waypoints.
    It also uses simple counting of missing imaging goals as the primary driver for heuristic value.
    """

    def __init__(self, task):
        """
        - Store goals related to (lk059nof ?objective ?resolution).
        - Build graph of waypoint connectivity from static facts (kxvuge0a).
        - Extract rovers and their stores from static facts (fa5tam1q).
        - Extract cameras and which objectives they support from (g52pa0bi).
        - Extract camera image types (xq9py0w8) per camera.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract goal imaging facts: tuples (objective, resolution)
        self.goal_images = []
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "lk059nof" and len(parts) == 3:
                self.goal_images.append((parts[1], parts[2]))

        # Build waypoint graph from kxvuge0a static facts
        self.waypoint_graph = {}
        for fact in self.static:
            if match(fact, "kxvuge0a", "*", "*"):
                _, from_wp, to_wp = get_parts(fact)
                self.waypoint_graph.setdefault(from_wp, set()).add(to_wp)

        # Extract rovers - from distint rovers in predicates with cdmmbhnd type
        self.rovers = set()
        # Rovers' stores: map store -> rover
        self.rover_stores = {}
        for fact in self.static:
            if match(fact, "fa5tam1q", "*", "*"):
                # (fa5tam1q store rover)
                _, store, rover = get_parts(fact)
                self.rover_stores[store] = rover
                self.rovers.add(rover)

        # Extract cameras and the objectives they can image (g52pa0bi camera objective)
        self.cameras_objectives = {}
        for fact in self.static:
            if match(fact, "g52pa0bi", "*", "*"):
                _, camera, objective = get_parts(fact)
                self.cameras_objectives.setdefault(camera, set()).add(objective)

        # Extract camera image types (xq9py0w8 camera resolution)
        self.camera_resolutions = {}
        for fact in self.static:
            if match(fact, "xq9py0w8", "*", "*"):
                _, camera, resolution = get_parts(fact)
                self.camera_resolutions.setdefault(camera, set()).add(resolution)

        # Precompute waypoint shortest paths with BFS to speed up heuristic
        self._precompute_shortest_paths()

    def _precompute_shortest_paths(self):
        """Compute and store shortest distances between all waypoints."""
        # BFS from each waypoint to all reachable waypoints
        self.shortest_paths = {}  # {from_wp: {to_wp: dist, ...}, ...}
        for start in self.waypoint_graph:
            visited = {start: 0}
            queue = [start]
            while queue:
                current = queue.pop(0)
                current_dist = visited[current]
                for neighbor in self.waypoint_graph.get(current, ()):
                    if neighbor not in visited:
                        visited[neighbor] = current_dist + 1
                        queue.append(neighbor)
            self.shortest_paths[start] = visited

    def _distance(self, from_wp, to_wp):
        """
        Return shortest path distance between waypoints,
        or a large cost if no path exists.
        """
        if from_wp == to_wp:
            return 0
        return self.shortest_paths.get(from_wp, {}).get(to_wp, 1000)

    def __call__(self, node):
        """
        For each imaging goal not yet satisfied in the current state,
        estimate minimal cost to achieve it by any rover, summing all.

        Estimation includes:
          - Moving rover from current location to imaging location (distance weight).
          - Preparing and having correct data store state (penalty).
          - Imaging action cost (1 action assumed).
        """
        state = node.state

        # Quickly return 0 if goal reached
        if self.goals <= state:
            return 0

        # Extract relevant current predicates:
        # - rover current locations (ilkg22gp rover waypoint)
        rover_locations = {}
        for fact in state:
            if match(fact, "ilkg22gp", "*", "*"):
                _, rover, wp = get_parts(fact)
                rover_locations[rover] = wp

        # - rovers with store ready or not (pmcrx641 = store ready, zdfck87b = store busy)
        store_ready = set()
        store_busy = set()
        for fact in state:
            if match(fact, "pmcrx641", "*"):
                _, store = get_parts(fact)
                store_ready.add(store)
            elif match(fact, "zdfck87b", "*"):
                _, store = get_parts(fact)
                store_busy.add(store)

        # - cameras assigned to rovers (uss6vmx6 camera rover)
        camera_rover = {}
        for fact in state:
            if match(fact, "uss6vmx6", "*", "*"):
                _, camera, rover = get_parts(fact)
                camera_rover[camera] = rover

        # - known imaging achievements l3943stc (rover objective resolution)
        achieved_images = set()
        for fact in state:
            if match(fact, "l3943stc", "*", "*", "*"):
                _, rover, objective, resolution = get_parts(fact)
                achieved_images.add((objective, resolution))

        total_cost = 0

        for objective, res in self.goal_images:
            # Skip if already achieved or lkn059nof goal fact true in state
            if ((objective, res) in achieved_images
                    or f"(lk059nof {objective} {res})" in state):
                continue

            # Candidate rovers are those that have a camera that can image this objective at this resolution
            candidate_rovers = []
            for camera, objs in self.cameras_objectives.items():
                if objective in objs and res in self.camera_resolutions.get(camera, set()):
                    rover = camera_rover.get(camera)
                    if rover is not None:
                        candidate_rovers.append((rover, camera))

            if not candidate_rovers:
                # No rover-camera to produce this image; assign high cost to discourage
                total_cost += 50
                continue

            # For each rover candidate, compute a rough cost:
            min_cost_for_goal = 1000

            for rover, camera in candidate_rovers:
                # Rover current location
                rover_wp = rover_locations.get(rover)
                if rover_wp is None:
                    # Rover location unknown; assume high cost
                    rover_to_obj_cost = 20
                else:
                    # Find waypoints from which the rover can image the objective
                    # The domain indicates imaging happens if rover is at waypoints connected appropriately.
                    # We use static connectivity (xuwncpn8 rover from_wp to_wp), and objective location (wecz86om objective wp).
                    # So check all waypoints from which rover can move to waypoint of objective; approximate by distance rovers wp->objective wp

                    # Find objective waypoints
                    obj_waypoints = set()
                    for fact in self.static:
                        if match(fact, "wecz86om", objective, "*"):
                            _, _, obj_wp = get_parts(fact)
                            obj_waypoints.add(obj_wp)

                    # If no known objective waypoint, assume high cost
                    if not obj_waypoints:
                        rover_to_obj_cost = 20
                    else:
                        # Find minimal distance from rover current wp to any objective wp
                        dist = min(self._distance(rover_wp, obj_wp) for obj_wp in obj_waypoints)
                        # Add 1 for moving onto the objective wp if needed
                        rover_to_obj_cost = dist + 1

                # Check store of rover and its readiness
                # Find store for this rover
                rover_store = None
                for store, rv in self.rover_stores.items():
                    if rv == rover:
                        rover_store = store
                        break

                store_penalty = 0
                if rover_store is not None:
                    if rover_store in store_busy:
                        # Need to toggle to ready: add cost
                        store_penalty += 2
                    elif rover_store not in store_ready:
                        store_penalty += 3  # cost to ready up store

                # Add imaging action cost (assumed 1)
                imaging_cost = 1

                # Sum components for this rover
                estimated_cost = rover_to_obj_cost + store_penalty + imaging_cost
                min_cost_for_goal = min(min_cost_for_goal, estimated_cost)

            total_cost += min_cost_for_goal

        # Heuristic should never return zero unless goal is reached
        return max(total_cost, 1)
