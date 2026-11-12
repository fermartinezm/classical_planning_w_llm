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

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal
    state by focusing on the main goals of the domain that are about achieving 
    the `(lk059nof objective resolution)` predicates for all objective-resolution pairs
    in the goal, where `resolution` is one of the image types (like high_res, low_res, colour).
    The heuristic systematically accounts for achieving these observations by:
      - Ensuring a rover is positioned at a suitable waypoint,
      - Ensuring a camera has taken the picture of the objective with the required resolution,
      - Inferring necessary moves of rovers between waypoints,
      - Handling store status (e.g. `pmcrx641` / `zdfck87b`) to enable camera usage,
      - Considering only the minimal cost to fulfill all unresolved image objectives.
    
    Assumptions
    -----------
    - Achieving `(lk059nof obj resolution)` means an image of a certain resolution is obtained of the objective.
    - To obtain an image, a rover must be at a waypoint related to the objective,
      a camera must be assigned to the rover, and the camera must have the image stored in the store.
    - Moving rovers between waypoints is done via `(iluwncpn8 ...)` edges (from static).
    - The camera must first "take" the image to place the required `(l3943stc rover objective resolution)` fact,
      after which `(lk059nof objective resolution)` is achieved.
    - There is a tradeoff of camera assignments to rovers and movement costs, approximated by counting missing facts.
    - The heuristic is not admissible but aims to roughly estimate the minimal necessary actions to reach the goals.

    Heuristic Initialization
    ------------------------
    - Parse goal facts to find all goals of the form `(lk059nof objective resolution)`.
    - Extract static facts for:
        * rover connectivity: `(xuwncpn8 rover waypoint1 waypoint2)` edges.
        * waypoint adjacencies: `(kxvuge0a waypoint other_waypoint)` edges.
        * for each rover, its store (`fa5tam1q store rover`) and store status.
        * facts indicating which cameras are assignable to which rovers (`uss6vmx6 camera rover`).
        * facts describing camera->objective associations `(g52pa0bi camera objective)`.
    - Precompute an approximation of shortest distances between waypoints for rover movement.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify unmet goals: for each goal `(lk059nof obj res)` not yet in the state,
       try to estimate the cost to achieve it.
    2. For each such objective-resolution pair:
       - Check if any rover currently holds `(l3943stc rover obj res)`:
         if yes, then only minimal further action may be needed.
       - Otherwise, estimate cost to achieve `(l3943stc rover obj res)`:
         * The rover must be at the waypoint where the objective is located (`wecz86om obj waypoint`).
         * The rover may need to move from its current position to that waypoint.
         * The rover's store must be enabled to allow picture taking (`pmcrx641` vs `zdfck87b`).
         * The camera must be assigned to the rover.
       - Summation of estimated moves and status changes gives action cost.
    3. Sum costs for all unmet objectives.
    4. The heuristic value is 0 if all goals are satisfied.
    """

    def __init__(self, task):
        self.goals = task.goals  # Goal facts as frozenset of strings.
        self.static = task.static  # Static facts as frozenset of strings.

        # Extract goals of form (lk059nof obj resolution)
        self.goal_objectives = []
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "lk059nof" and len(parts) == 3:
                obj = parts[1]
                resolution = parts[2]
                self.goal_objectives.append((obj, resolution))

        # Extract static connectivity between waypoints for rovers. Using (xuwncpn8 rover wp1 wp2)
        # But since domain is static info, we build a graph of waypoints connected by kxvuge0a (bidirectional edges)
        self.waypoint_graph = {}
        for fact in self.static:
            if match(fact, "kxvuge0a", "*", "*"):
                _, wp1, wp2 = get_parts(fact)
                if wp1 not in self.waypoint_graph:
                    self.waypoint_graph[wp1] = set()
                if wp2 not in self.waypoint_graph:
                    self.waypoint_graph[wp2] = set()
                self.waypoint_graph[wp1].add(wp2)
                self.waypoint_graph[wp2].add(wp1)

        # Extract rovers and their stores:
        # (fa5tam1q roverstore rover) -- links stores to rovers
        self.roverstores = {}
        for fact in self.static:
            if match(fact, "fa5tam1q", "*", "*"):
                _, store, rover = get_parts(fact)
                self.roverstores[rover] = store

        # Extract cameras and which rovers they can be assigned to via (uss6vmx6 camera rover)
        self.camera2rovers = {}
        for fact in self.static:
            if match(fact, "uss6vmx6", "*", "*"):
                _, camera, rover = get_parts(fact)
                self.camera2rovers.setdefault(camera, set()).add(rover)

        # Extract camera-objective pairs: (g52pa0bi camera objective)
        self.camera_objectives = {}
        for fact in self.static:
            if match(fact, "g52pa0bi", "*", "*"):
                _, camera, objective = get_parts(fact)
                self.camera_objectives.setdefault(objective, set()).add(camera)

        # Extract which objective is at which waypoint: (wecz86om objective waypoint)
        self.objective_waypoints = {}
        for fact in self.static:
            if match(fact, "wecz86om", "*", "*"):
                _, obj, wp = get_parts(fact)
                self.objective_waypoints[obj] = wp

        # Compute approximate shortest distances between waypoints using BFS for fast heuristic estimation
        self._precompute_waypoint_distances()

    def _precompute_waypoint_distances(self):
        """Compute shortest path distances between all waypoints using BFS."""
        self.wp_distances = {}
        for start_wp in self.waypoint_graph:
            dist = {start_wp: 0}
            queue = [start_wp]
            while queue:
                current = queue.pop(0)
                for neighbor in self.waypoint_graph[current]:
                    if neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)
            self.wp_distances[start_wp] = dist

    def __call__(self, node):
        state = node.state

        # Goal reached test: heuristic is zero if all goals satisfied
        if self.goals <= state:
            return 0

        # Helper to find which rovers hold l3943stc fact with objective and resolution:
        # (l3943stc ?rover ?objective ?resolution)
        # Also record rovers' current waypoint positions from (ilkg22gp rover waypoint)
        rovers_with_picture = {}
        rover_positions = {}
        # Also track store status of rovers' stores: store can be available (pmcrx641) or empty (zdfck87b)
        store_status = {}
        # Track which cameras are currently assigned to which rovers: (v25mqvwj camera rover)
        assigned_cameras = {}
        # Cameras held by rovers in the state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "l3943stc" and len(parts) == 4:
                rover, obj, resolution = parts[1], parts[2], parts[3]
                rovers_with_picture[(obj, resolution)] = rover
            elif parts[0] == "ilkg22gp" and len(parts) == 3:
                rover, wp = parts[1], parts[2]
                rover_positions[rover] = wp
            elif parts[0] == "pmcrx641" and len(parts) == 2:
                store = parts[1]
                store_status[store] = "available"
            elif parts[0] == "zdfck87b" and len(parts) == 2:
                store = parts[1]
                store_status[store] = "empty"
            elif parts[0] == "v25mqvwj" and len(parts) == 3:
                camera, rover = parts[1], parts[2]
                assigned_cameras[camera] = rover

        # Heuristic sum to accumulate cost
        total_cost = 0

        # For every goal objective-resolution, estimate missing cost
        for (obj, resolution) in self.goal_objectives:
            # If goal already satisfied (lk059nof obj resolution) in state, no cost
            goal_fact = f"(lk059nof {obj} {resolution})"
            if goal_fact in state:
                continue  # already achieved goal

            # If picture is already taken (l3943stc rover obj resolution), cost minimal
            if (obj, resolution) in rovers_with_picture:
                # Only remaining action may be to finalize lk059nof fact, but ignore since small impact
                total_cost += 1  # at least one action needed to finalize
                continue

            # Else need to produce l3943stc fact:
            # Find objective's waypoint
            if obj not in self.objective_waypoints:
                # Defensive fallback: if no known waypoint of objective, assume large cost
                total_cost += 10
                continue

            target_wp = self.objective_waypoints[obj]

            # Find cameras that can take picture of obj
            possible_cameras = self.camera_objectives.get(obj, set())
            if not possible_cameras:
                total_cost += 10  # no camera can take picture, large cost
                continue

            # For each camera, find rovers it can be assigned to
            min_cost_for_obj = float('inf')
            for camera in possible_cameras:
                possible_rovers = self.camera2rovers.get(camera, set())
                if not possible_rovers:
                    continue  # no rover can use this camera

                # For each rover, compute estimated cost:
                for rover in possible_rovers:
                    cost = 0

                    # Distance rover current position to objective waypoint (rover must be there)
                    rover_pos = rover_positions.get(rover)
                    if rover_pos is None:
                        # Rover position unknown, assume high cost
                        cost += 5
                    else:
                        if rover_pos != target_wp:
                            dist = self.wp_distances.get(rover_pos, {}).get(target_wp, None)
                            if dist is None:
                                # No path known, assume large cost
                                cost += 5
                            else:
                                cost += dist  # each move approximately 1 action

                    # Store status: rover's store must be available (pmcrx641) not empty (zdfck87b)
                    store = self.roverstores.get(rover)
                    if store is None:
                        # Unknown store, assume 1 action to fix
                        cost += 1
                    else:
                        status = store_status.get(store, "empty")
                        if status == "empty":
                            cost += 2  # one to enable store (o64yr71v), one camera usage cost

                    # Camera must be assigned to rover (v25mqvwj camera rover)
                    assigned_rover = assigned_cameras.get(camera)
                    if assigned_rover != rover:
                        cost += 2  # one action to assign camera, one to unassign if needed

                    # Add 1 cost to take the photo (azu4u5xz or cp8otkvs / wyhd5xzi etc.)
                    cost += 1

                    # Add 1 cost to finalize lk059nof fact by operator a7xj6iq1 or bypa1zm7 etc.
                    cost += 1

                    # Keep minimal cost over options
                    if cost < min_cost_for_obj:
                        min_cost_for_obj = cost

            # Add minimal estimated cost for this objective-resolution pair
            if min_cost_for_obj == float('inf'):
                # No possible way found, assign large cost
                min_cost_for_obj = 10

            total_cost += min_cost_for_obj

        return total_cost
