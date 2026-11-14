from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    
    :param fact: The fact as a string, e.g. "(ilkg22gp rover1 waypoint1)"
    :param args: The pattern elements. Use '*' as wildcard.
    :return: True if fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary:
    This heuristic estimates the number of actions required to achieve all the goal facts
    related to the `(lk059nof objectiveX resolution)` predicates in the secret_domain.
    The domain revolves around rovers (cdmmbhnd) at waypoints (gwg9gujk) capturing images
    with cameras (f3vbbxzk) of various objectives (zi8ouvk1) in different resolutions (kad6swsx).
    The key goal is to have all `(lk059nof objective resolution)` facts true for required objectives and resolutions.

    Assumptions:
    - The goals are always of the form `(lk059nof objective resolution)`.
    - To get `(lk059nof objective resolution)` true:
      1. The rover must acquire the image data `(l3943stc rover objective resolution)` with a suitable camera.
      2. The rover must have transferred data from store (v4e7u9dr) and have processed goals tracked by `(chfbweyv waypoint)` predicates.
    - Rovers can move between waypoints if `(xuwncpn8 rover waypoint1 waypoint2)` and `(kxvuge0a waypoint1 waypoint2)` are true.
    - Cameras are associated with objectives via `(g52pa0bi camera objective)` and resolutions via `(xq9py0w8 camera resolution)`.
    - Data transport involves `pmcrx641` (store ready to send) and `zdfck87b` (store busy/processed).
    - Some conditions require waypoints marked with `(chfbweyv waypoint)`.

    Heuristic Initialization:
    - Extract the goal requirements as tuples (objective, resolution) from `(lk059nof ...)` goals.
    - Extract static connectivity between waypoints via `(kxvuge0a waypoint1 waypoint2)` facts, to compute shortest paths.
    - Extract which cameras can capture which objectives and which resolutions.
    - Extract rover-store relations `(fa5tam1q store rover)` and store states `(pmcrx641 store)`.
    - Precompute shortest path distances between waypoints using static connectivity for efficient distance estimation.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each goal `(lk059nof obj res)`, check if it is already achieved in the current state (heuristic contribution 0).
    2. If not achieved:
       a) Identify rovers that currently have `(l3943stc rover obj res)` (image data acquired) - no need to reacquire, else assume 1 action needed to capture image (simplification).
       b) Identify waypoint where this rover currently is `(ilkg22gp rover waypoint)`.
       c) Identify waypoints where the objective is located `(wecz86om obj waypoint)`.
       d) Estimate minimal movement cost for the rover to reach an appropriate waypoint to acquire the image if not already done (using shortest waypoint distance).
       e) Identify if the rover’s store is ready `(pmcrx641 store)` and associated `(fa5tam1q store rover)`. Account for data transfer actions, which seem needed to finally establish the `(lk059nof obj res)` fact.
       f) Account for data processing steps and the necessity of waypoints marked `(chfbweyv waypoint)` for completion.
    3. Sum these estimated costs over all goal `(lk059nof ...)` facts not yet achieved.
    4. Ensure heuristic is 0 if all goals are met.

    We simplify the heuristic by:
    - Counting the number of missing `(lk059nof obj res)` goals.
    - For each missing goal, estimate cost as:
      - 1 for capture if not present,
      - plus rover movement cost between its current waypoint and the objective's associated waypoints (minimum shortest path),
      - plus 1 for data transfer action,
      - plus 1 for possible processing/confirmation step,
    - Aggregate over all to guide search efficiently.

    This heuristic ignores complex ordering dependencies and synchronization details for efficiency but provides a reasonable estimate guiding rovers to capture required imagery and transfer data.
    """

    def __init__(self, task):
        self.goals_lk059nof = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "lk059nof" and len(parts) == 3:
                # goal (lk059nof objective resolution)
                self.goals_lk059nof.add((parts[1], parts[2]))

        # Extract static connectivity between waypoints for shortest path computation.
        # Graph: waypoint -> set of connected waypoints
        self.graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "kxvuge0a" and len(parts) == 3:
                w1, w2 = parts[1], parts[2]
                self.graph.setdefault(w1, set()).add(w2)

        # Precompute all pairs shortest path distances between waypoints using BFS
        self.waypoint_distances = {}
        waypoints = list(self.graph.keys())
        for start in waypoints:
            dist = {start: 0}
            queue = [start]
            while queue:
                current = queue.pop(0)
                for neigh in self.graph.get(current, []):
                    if neigh not in dist:
                        dist[neigh] = dist[current] + 1
                        queue.append(neigh)
            self.waypoint_distances[start] = dist

        # Map objectives to waypoints where located (wecz86om objective waypoint)
        self.objective_locations = {}
        for fact in task.static | task.initial_state:
            parts = get_parts(fact)
            if parts[0] == "wecz86om" and len(parts) == 3:
                obj, wp = parts[1], parts[2]
                self.objective_locations.setdefault(obj, set()).add(wp)

        # Capture cameras that can take images of each objective: (g52pa0bi camera objective)
        self.camera_objectives = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "g52pa0bi" and len(parts) == 3:
                cam, obj = parts[1], parts[2]
                self.camera_objectives.setdefault(obj, set()).add(cam)

        # Camera supports which resolutions: (xq9py0w8 camera resolution)
        self.camera_resolutions = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "xq9py0w8" and len(parts) == 3:
                cam, res = parts[1], parts[2]
                self.camera_resolutions.setdefault(cam, set()).add(res)

        # Store to rover mapping (fa5tam1q store rover)
        self.store_to_rover = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "fa5tam1q" and len(parts) == 3:
                store, rover = parts[1], parts[2]
                self.store_to_rover[store] = rover

    def __call__(self, node):
        state = node.state

        # Quick check: if all (lk059nof ...) goals are reached, return 0
        if self.goals_lk059nof <= {
            (get_parts(f)[1], get_parts(f)[2]) for f in state if f.startswith("(lk059nof ")
        }:
            return 0

        # Extract current rover locations: (ilkg22gp rover waypoint)
        rover_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ilkg22gp":
                rover, wp = parts[1], parts[2]
                rover_locations[rover] = wp

        # Extract which rovers have which stores (fa5tam1q store rover from static already stored)
        # Extract store ready states (pmcrx641 store)
        stores_ready = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "pmcrx641":
                stores_ready.add(parts[1])

        # Extract which rovers currently have images for which objective and resolution:
        # (l3943stc rover objective resolution)
        rover_images = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "l3943stc" and len(parts) == 4:
                rover, obj, res = parts[1], parts[2], parts[3]
                rover_images.setdefault(rover, set()).add((obj, res))

        # Extract waypoints where data processing has been completed: (chfbweyv waypoint)
        processed_waypoints = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "chfbweyv":
                processed_waypoints.add(parts[1])

        # For each (lk059nof obj res) goal not yet achieved, estimate costs
        cost = 0
        achieved = {
            (get_parts(f)[1], get_parts(f)[2]) for f in state if f.startswith("(lk059nof ")
        }
        missing_goals = self.goals_lk059nof - achieved

        for obj, res in missing_goals:
            # 1) Check if any rover already has image (obj,res), cost 0 for capture else 1
            image_had = False
            capturing_rovers = []
            for r, images in rover_images.items():
                if (obj, res) in images:
                    image_had = True
                    capturing_rovers.append(r)
            capturing_cost = 0 if image_had else 1

            # 2) Estimate minimal rover movement cost to objective waypoint:
            # Find the nearest rover waypoint among rovers that can capture this image, else any rover.
            min_move_cost = None

            # Which rovers can capture this obj-res? 
            # First find cameras that capture obj and res
            cams = self.camera_objectives.get(obj, set())
            cams_matching_res = {c for c in cams if res in self.camera_resolutions.get(c, set())}
            # Which rovers have those cameras? From (uss6vmx6 camera rover) facts in initial/static
            camera_rover_map = {}
            for fact in node.task.static | node.state:
                parts = get_parts(fact)
                if parts[0] == "uss6vmx6" and len(parts) == 3:
                    camera_rover_map.setdefault(parts[2], set()).add(parts[1])

            candidate_rovers_camera = set()
            for r, cams_set in camera_rover_map.items():
                # r is rover, cams_set cameras assigned? Actually parts are (uss6vmx6 camera rover)
                # Special case: key is rover, value set of cameras it has.
                pass
            # Actually the fact: (uss6vmx6 camera rover) means camera attached to rover
            # Reverse mapping camera -> rover:
            camera_to_rovers = {}
            for fact in node.task.static | node.state:
                parts = get_parts(fact)
                if parts[0] == "uss6vmx6" and len(parts) == 3:
                    cam_, rov_ = parts[1], parts[2]
                    camera_to_rovers.setdefault(cam_, set()).add(rov_)
            # rovers that have cameras supporting obj & res:
            candidate_rovers_camera = set()
            for cam in cams_matching_res:
                candidate_rovers_camera.update(camera_to_rovers.get(cam, set()))

            # Now compute min movement from rover current waypoint to any objective waypoint
            obj_wps = self.objective_locations.get(obj, set())
            if not obj_wps:
                # Defensive fallback: if no objective location known, 0 movement cost
                min_move_cost = 0
            else:
                rovers_to_consider = capturing_rovers if capturing_rovers else candidate_rovers_camera
                if not rovers_to_consider:
                    # fallback: consider all rovers, just minimize distance
                    rovers_to_consider = rover_locations.keys()
                distances_found = []
                for rov in rovers_to_consider:
                    rov_wp = rover_locations.get(rov, None)
                    if not rov_wp:
                        continue
                    dist_map = self.waypoint_distances.get(rov_wp, {})
                    min_dist = min((dist_map.get(obj_wp, 1000) for obj_wp in obj_wps), default=1000)
                    distances_found.append(min_dist)
                if distances_found:
                    min_move_cost = min(distances_found)
                else:
                    # No rover location known => no movement estimated
                    min_move_cost = 0

            # 3) Check if rover's store is ready to send data (from store to rover map)
            # Count 1 action for data transfer if store ready, else 2 (send + ready)
            # Use any store associated with rovers that are capturing or candidate_rovers
            # Just check stores for candidate rovers, else count 2 to be safe
            any_store_ready = False
            for store, rover in self.store_to_rover.items():
                if (not candidate_rovers_camera or rover in candidate_rovers_camera) and store in stores_ready:
                    any_store_ready = True
                    break
            data_transfer_cost = 1 if any_store_ready else 2

            # 4) Processing cost: at least 1 for processing or waypoint confirmation (assume 1)
            processing_cost = 1

            # Aggregate cost for this goal
            goal_cost = capturing_cost + min_move_cost + data_transfer_cost + processing_cost

            cost += goal_cost

        return cost if cost > 0 else 1  # Prefer at least 1 if not goal reached to break ties
