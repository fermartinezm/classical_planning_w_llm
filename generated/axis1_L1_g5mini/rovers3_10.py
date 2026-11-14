from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact string: '(pred a b)' -> ['pred','a','b']"""
    return fact[1:-1].split()

def match(fact, *pattern):
    """True if fact components match the given pattern (supports '*' wildcards)."""
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (rover-like) domain.
    The primary goals in this domain are of the form (lk059nof <objective> <res-type>).
    To satisfy such a goal typically requires:
      - a camera compatible with the objective and the resource,
      - a rover at a waypoint associated with that objective,
      - mounting the camera on the rover (jx1xpj7f),
      - taking the image with the mounted camera (tkek73dx), and
      - processing the taken image to produce the final goal fact (bypa1zm7).
    This heuristic estimates the number of actions remaining by computing, for
    each missing lk059nof goal, a compact cost composed of:
      movement steps for a rover to reach a waypoint for the objective +
      mount (if needed) + take image + process image.
    The heuristic sums the (independently minimal) costs for all missing goal facts.

    Assumptions
    -----------
    - Static facts define: which waypoints an objective can be observed from
      (wecz86om), which cameras support which objectives (g52pa0bi),
      which cameras support which resource types (xq9py0w8),
      and which cameras can mount on which rovers (uss6vmx6).
    - Rover movement is possible only along edges listed as xuwncpn8 facts in the
      static facts (these are rover-specific directed edges).
    - Mounting a camera (jx1xpj7f) is required before taking an image unless
      the state already contains a (v25mqvwj camera rover) fact for the
      correct camera mounted on the rover that will perform the imaging.
    - If l3943stc (taken-image) facts for an (objective, res-type) already
      exist in the current state, we optimistically assume only the processing
      action (bypa1zm7) remains (i.e., cost 1), since that image already
      satisfies tkek73dx's production step.
    - Heuristic does not attempt to globally optimize reuse of rovers/cameras
      between multiple goals (it combines minimal independent costs). This
      keeps it efficient while providing informative guidance.

    Heuristic Initialization
    ------------------------
    In the constructor we extract the following from the task.static facts:
      - adjacency per rover from xuwncpn8 facts:
          self.adjacency[rover][waypoint] = set(neighbor_waypoints)
      - objective -> set(waypoints) from wecz86om facts
      - camera -> set(objectives) from g52pa0bi facts
      - camera -> set(resources) from xq9py0w8 facts
      - camera -> set(rover) mountability from uss6vmx6 facts
    We also store the declarative goals (task.goals) for quick checking.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
      1. If all goal facts (task.goals) are present in the state, return 0.
      2. Parse current dynamic facts:
         - current rover positions: ilkg22gp rover waypoint
         - currently mounted cameras: v25mqvwj camera rover
         - already taken images: l3943stc rover objective res  (used to short-circuit)
      3. For each goal fact (lk059nof objective res) that is not present:
         a. If there exists any l3943stc * * objective res in the state:
            - We conservatively assume only the processing action remains: cost = 1.
         b. Otherwise, evaluate candidate (rover, camera, waypoint) combinations:
            - Waypoints are taken from static wecz86om facts for that objective.
            - For each rover that has a known current location in the state and
              for which the static adjacency (xuwncpn8) graph is available:
                i. Compute shortest-path distance (BFS) from rover's current waypoint
                   to any waypoint associated with the objective (distance measured
                   in number of azu4u5xz moves).
               ii. For each camera that (statically) supports both the objective
                   and the resource type, and that can be mounted on that rover:
                    - Determine mount_cost:
                        0 if the camera is already mounted on that same rover in state (v25mqvwj)
                        1 if the camera is currently not mounted on any rover (available)
                        large penalty if camera is mounted on a different rover (treated as unavailable)
                    - take_cost = 1 (tkek73dx)
                    - process_cost = 1 (bypa1zm7)
                    - candidate cost = movement_distance + mount_cost + take_cost + process_cost
            - The minimal candidate cost over all (rover, camera) pairs is taken for this goal.
            - If no feasible candidate is found, we assign a large finite penalty cost so the heuristic remains finite.
      4. Sum the per-goal minimal costs to form the returned heuristic value.
      5. Return the total estimated cost.

    Notes on design tradeoffs:
    - The heuristic intentionally keeps per-goal costs independent (sum), which
      can overestimate when a single rover/camera can service multiple goals in
      one route. However, the estimate remains informative and cheap to compute.
    - The heuristic is not admissible but favors states that have rovers near
      required waypoints and with compatible cameras already mounted.
    """

    def __init__(self, task):
        # Goals (set of strings)
        self.goals = task.goals
        static_facts = task.static

        # adjacency per rover: rover -> {waypoint: set(neighbors)}
        self.adjacency = {}  # rover -> waypoint -> set(neigh)
        # wecz86om: objective -> set(waypoints)
        self.objective_waypoints = {}
        # camera -> set(objectives)
        self.camera_objectives = {}
        # camera -> set(resources)
        self.camera_resources = {}
        # camera -> set(rover) mountable
        self.camera_mountable = set()

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "xuwncpn8":
                # (xuwncpn8 rover from to)
                rover, frm, to = parts[1], parts[2], parts[3]
                self.adjacency.setdefault(rover, {}).setdefault(frm, set()).add(to)
            elif pred == "wecz86om":
                # (wecz86om objective waypoint)
                obj, wp = parts[1], parts[2]
                self.objective_waypoints.setdefault(obj, set()).add(wp)
            elif pred == "g52pa0bi":
                # (g52pa0bi camera objective)
                cam, obj = parts[1], parts[2]
                self.camera_objectives.setdefault(cam, set()).add(obj)
            elif pred == "xq9py0w8":
                # (xq9py0w8 camera resource)
                cam, res = parts[1], parts[2]
                self.camera_resources.setdefault(cam, set()).add(res)
            elif pred == "uss6vmx6":
                # (uss6vmx6 camera rover)
                cam, rover = parts[1], parts[2]
                self.camera_mountable.add((cam, rover))

        # large finite penalty for impossible/unavailable choices
        self.INF_COST = 1000

    def _shortest_distance(self, rover, start, targets):
        """
        BFS shortest directed distance from start to any node in targets
        using the rover-specific adjacency. Returns integer distance (0 if start in targets)
        or None if unreachable / no adjacency exists for this rover.
        """
        if start in targets:
            return 0
        if rover not in self.adjacency:
            return None
        adj = self.adjacency[rover]
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, d = q.popleft()
            neighbors = adj.get(node, ())
            for nb in neighbors:
                if nb in visited:
                    continue
                if nb in targets:
                    return d + 1
                visited.add(nb)
                q.append((nb, d + 1))
        return None

    def __call__(self, node):
        state = node.state

        # Quick goal check: return 0 if all goals satisfied
        if self.goals <= state:
            return 0

        # Parse dynamic facts from the current state
        rover_at = {}  # rover -> waypoint
        mounted = set()  # (camera, rover)
        taken_images = set()  # tuples (objective, res) from l3943stc facts (we only need obj and res)
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "ilkg22gp":
                # (ilkg22gp rover waypoint)
                rover, wp = parts[1], parts[2]
                rover_at[rover] = wp
            elif pred == "v25mqvwj":
                # (v25mqvwj camera rover)
                cam, rover = parts[1], parts[2]
                mounted.add((cam, rover))
            elif pred == "l3943stc":
                # (l3943stc rover objective res)
                # store just (objective, res)
                if len(parts) >= 4:
                    _, obj, res = parts[1], parts[2], parts[3]
                    taken_images.add((obj, res))

        total_cost = 0

        # For each goal (lk059nof objective res) that is missing, compute minimal cost
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] != "lk059nof":
                # not a predicate we reason about; skip (shouldn't happen in this domain)
                continue
            obj, res = parts[1], parts[2]
            if goal in state:
                continue  # already satisfied

            # If a taken image (l3943stc) for this objective/res exists, only processing remains
            if (obj, res) in taken_images:
                total_cost += 1  # bypa1zm7
                continue

            # Otherwise, we need: move rover to a waypoint for this objective + mount (maybe) + take + process
            waypoints = self.objective_waypoints.get(obj, set())
            if not waypoints:
                # no static info relating objective to waypoints -> penalize heavily
                total_cost += self.INF_COST
                continue

            best_cost = self.INF_COST
            # Try all rovers available in the current state
            for rover, start_wp in rover_at.items():
                # compute distance to any waypoint observing the objective
                dist = self._shortest_distance(rover, start_wp, waypoints)
                if dist is None:
                    continue  # rover cannot reach any relevant waypoint

                # For this rover, consider cameras that support both objective and resource
                for cam, objs in self.camera_objectives.items():
                    if obj not in objs:
                        continue
                    if res not in self.camera_resources.get(cam, set()):
                        continue
                    # Check mountability of this camera on this rover
                    if (cam, rover) not in self.camera_mountable:
                        continue

                    # Determine mount cost:
                    if (cam, rover) in mounted:
                        mount_cost = 0  # already mounted on this rover
                    else:
                        # If camera is currently mounted on a different rover, treat as unavailable.
                        cam_currently_mounted_elsewhere = any(
                            (cam == mcam and mrover != rover) for (mcam, mrover) in mounted
                        )
                        if cam_currently_mounted_elsewhere:
                            # We conservatively skip this camera (not available)
                            continue
                        else:
                            mount_cost = 1  # needs mounting via jx1xpj7f

                    # take cost and process cost
                    take_cost = 1  # tkek73dx
                    process_cost = 1  # bypa1zm7

                    candidate_cost = dist + mount_cost + take_cost + process_cost
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost

            # If no rover-camera pair found, penalize but keep finite
            if best_cost == self.INF_COST:
                total_cost += self.INF_COST
            else:
                total_cost += best_cost

        return total_cost