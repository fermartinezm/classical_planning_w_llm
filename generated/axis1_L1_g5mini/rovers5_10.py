from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact string by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain ("cdmmbhnd") rover/camera-style domain.
    The domain requires rovers to move to waypoints, mount cameras, take images of objectives
    in particular modes, and then perform a final step that turns taken images into
    the goal facts. This heuristic estimates the number of actions required to achieve all
    remaining goal facts of the form (lk059nof <objective> <mode>).

    Assumptions
    -----------
    - Goals of interest are only the (lk059nof ...) predicate goals (these are the domain's
      image-delivery goals).
    - Taking an image and delivering it to the goal is the canonical way to achieve lk059nof.
    - Cameras, camera modes, objectives' waypoint positions, rover movement connectivity, and
      camera-rover associations are static (provided in `task.static`), and those are used to
      determine reachability and minimal move steps.
    - A small, fixed cost is used for non-move actions:
        * mounting a camera (jx1xpj7f) : cost 1
        * taking an image (tkek73dx)   : cost 1
        * converting a taken image -> goal (bypa1zm7) : cost 1
      Movement between adjacent waypoints (azu4u5xz) is counted as cost 1 per move.
    - Multiple goals are handled by summing per-goal estimates (no accounting for shared work).
      This is intentionally simple and admissibility is not required.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and index the static facts that are relevant to the estimate:
    - wecz86om <objective> <waypoint> : where each objective can be imaged from
    - e9bo95dc general <waypoint>      : waypoints usable for the final delivery precondition
    - kxvuge0a <wp1> <wp2>            : waypoint adjacency used to identify RL70M candidates
    - xuwncpn8 <rover> <wp_from> <wp_to>: rover-specific move edges (used to compute distances)
    - g52pa0bi <camera> <objective>   : which camera can image which objective
    - xq9py0w8 <camera> <mode>        : which camera supports which mode
    - uss6vmx6 <camera> <rover>       : which camera can be mounted on which rover
    - jy67mgmv <rover>                 : rovers eligible to perform mount/take steps
    The constructor also parses and stores the set of goal (objective, mode) pairs.
    Distances for rover movements are computed on demand with BFS per rover and cached.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If the state's goals are already satisfied, return 0.
    2. Parse the state to find:
       - current rover locations (ilkg22gp rover waypoint)
       - already taken images (l3943stc rover objective mode)
       - already achieved lk059nof facts (these reduce the number of remaining goals)
    3. For each remaining goal (lk059nof objective mode) compute a minimal estimated cost:
       a. If lk059nof is already in the state -> cost 0 for that goal.
       b. If an image l3943stc (for that objective and mode) already exists in the state (it has
          an associated rover R), then the remaining work is to run bypa1zm7:
            - Move rover R from its current location to any waypoint RL70M that has a kxvuge0a
              edge to a waypoint that satisfies e9bo95dc (general waypoint). Movement cost is
              distance(R_current -> RL70M). Then +1 for bypa1zm7.
       c. Otherwise (no image exists yet): the canonical sequence is:
            - Move some eligible rover to a waypoint from which the objective can be imaged
              (wecz86om objective -> image_waypoint).
            - Mount a suitable camera on that rover and take the image:
                * mount cost = 1 (jx1xpj7f)
                * take cost  = 1 (tkek73dx)
              A camera is suitable if:
                - it can image the objective (g52pa0bi)
                - it supports the required mode (xq9py0w8)
                - it can be mounted on the rover (uss6vmx6)
              The rover must also be eligible (jy67mgmv).
            - After taking the image, the rover may need to move to an RL70M as in step (b)
              and run bypa1zm7 (+1).
            - Total = distance(current -> image_waypoint) + 2 (mount + take) +
                      distance(image_waypoint -> RL70M) + 1 (bypa1zm7).
       d. Over all rovers and valid cameras, choose the minimal cost computed above. If no valid
          camera/route exists, treat the goal cost as a large finite number (this keeps the
          heuristic finite).
    4. The heuristic value is the sum of the minimal costs computed for each remaining lk059nof
       goal. (This simple additive scheme is not perfect but is efficient and effective for
       guiding greedy best-first search.)
    """

    def __init__(self, task):
        """Extract static facts and precompute indexes needed by the heuristic."""
        self.goals = set(task.goals)  # raw goal fact strings
        static = task.static

        # Maps and sets extracted from static facts
        self.wecz_obj_to_waypoints = defaultdict(set)  # objective -> set(waypoints)
        self.e9bo_waypoints = set()  # waypoints that satisfy (e9bo95dc general waypoint)
        self.kx_edges = defaultdict(set)  # wp -> set(neighbor_wp) from kxvuge0a
        # movement edges per rover from xuwncpn8: rover -> {wp_from: set(wp_to)}
        self.xuwn_edges = defaultdict(lambda: defaultdict(set))

        # camera -> objectives, camera -> modes, camera -> rovers
        self.camera_objectives = defaultdict(set)
        self.camera_modes = defaultdict(set)
        self.camera_rovers = defaultdict(set)

        # rovers allowed to mount/take images (jy67mgmv)
        self.rovers_with_mount = set()

        # Parse static facts
        for fact in static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "wecz86om":
                # (wecz86om objective waypoint)
                obj, wp = parts[1], parts[2]
                self.wecz_obj_to_waypoints[obj].add(wp)
            elif pred == "e9bo95dc":
                # (e9bo95dc general waypoint) - first arg is usually 'general'
                # store the waypoint (parts[2])
                if len(parts) >= 3:
                    wp = parts[2]
                    self.e9bo_waypoints.add(wp)
            elif pred == "kxvuge0a":
                # (kxvuge0a wp1 wp2)
                wp1, wp2 = parts[1], parts[2]
                self.kx_edges[wp1].add(wp2)
            elif pred == "xuwncpn8":
                # (xuwncpn8 rover wp_from wp_to)
                rover, wp_from, wp_to = parts[1], parts[2], parts[3]
                self.xuwn_edges[rover][wp_from].add(wp_to)
            elif pred == "g52pa0bi":
                # (g52pa0bi camera objective)
                cam, obj = parts[1], parts[2]
                self.camera_objectives[obj].add(cam)
            elif pred == "xq9py0w8":
                # (xq9py0w8 camera mode)
                cam, mode = parts[1], parts[2]
                self.camera_modes[cam].add(mode)
            elif pred == "uss6vmx6":
                # (uss6vmx6 camera rover)
                cam, rover = parts[1], parts[2]
                self.camera_rovers[cam].add(rover)
            elif pred == "jy67mgmv":
                # (jy67mgmv rover)
                rover = parts[1]
                self.rovers_with_mount.add(rover)

        # Precompute RL70M candidate waypoints:
        # RL70M is a waypoint wp such that exists kxvuge0a wp -> v where v is an e9bo95dc waypoint.
        self.rl_candidates = set()
        for wp, neighbors in self.kx_edges.items():
            if any(neigh in self.e9bo_waypoints for neigh in neighbors):
                self.rl_candidates.add(wp)

        # Parse and store the set of goal (objective, mode) pairs for lk059nof
        self.goal_obj_mode_pairs = []
        for g in self.goals:
            parts = get_parts(g)
            if not parts:
                continue
            if parts[0] == "lk059nof" and len(parts) >= 3:
                obj, mode = parts[1], parts[2]
                self.goal_obj_mode_pairs.append((obj, mode))

        # BFS distance cache: (rover, start_wp) -> {wp: distance}
        self._dist_cache = {}

    def _bfs_distances(self, rover, start):
        """
        Return a dict mapping waypoints reachable from start to their shortest
        distance (number of moves) for a given rover using xuwncpn8 edges.
        Caches results in self._dist_cache.
        """
        key = (rover, start)
        if key in self._dist_cache:
            return self._dist_cache[key]

        adj = self.xuwn_edges.get(rover, {})
        dist = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nb in adj.get(cur, ()):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        self._dist_cache[key] = dist
        return dist

    def __call__(self, node):
        """
        Compute a domain-dependent heuristic estimate for 'node.state'.

        The heuristic is the sum over all remaining lk059nof goals of the minimal estimated
        cost to achieve each one, where cost accounts for movement steps and a small fixed
        number for each required non-move action (mount, take, deliver).
        """
        state = node.state

        # 1) If goal already reached, heuristic is 0
        # The Task.goal_reached check is not available here; we simulate by checking all goals.
        if self.goals <= state:
            return 0

        # Parse state: rover positions, existing images (l3943stc), existing lk059nofs
        rover_positions = {}  # rover -> waypoint
        existing_images = set()  # set of tuples (objective, mode, rover) from l3943stc
        achieved_lk = set()  # set of (obj, mode) already in state

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "ilkg22gp" and len(parts) >= 3:
                rover, wp = parts[1], parts[2]
                rover_positions[rover] = wp
            elif pred == "l3943stc" and len(parts) >= 4:
                rover, obj, mode = parts[1], parts[2], parts[3]
                existing_images.add((obj, mode, rover))
            elif pred == "lk059nof" and len(parts) >= 3:
                obj, mode = parts[1], parts[2]
                achieved_lk.add((obj, mode))

        total_cost = 0
        LARGE_COST = 1000  # finite but large for impossible/unavailable goals

        # For each goal (objective, mode), estimate minimal cost
        for obj, mode in self.goal_obj_mode_pairs:
            if (obj, mode) in achieved_lk:
                continue  # goal already satisfied in this state

            best_cost_for_goal = LARGE_COST

            # Option A: An image for this (obj,mode) already exists (l3943stc), so we only need bypa1zm7
            # bypa1zm7 requires the same rover that produced l3943stc to be at some RL70M.
            for (i_obj, i_mode, img_rover) in existing_images:
                if i_obj == obj and i_mode == mode:
                    # Must move img_rover from its current position to any RL70M candidate and apply bypa1zm7 (+1)
                    cur_wp = rover_positions.get(img_rover)
                    if cur_wp is None:
                        # Rover position unknown in state, skip this image (can't estimate)
                        continue
                    distances = self._bfs_distances(img_rover, cur_wp)
                    # minimal distance to any RL70M
                    min_dist = min((distances.get(rwp, float("inf")) for rwp in self.rl_candidates), default=float("inf"))
                    if min_dist != float("inf"):
                        cost = min_dist + 1  # move + bypa1zm7
                        best_cost_for_goal = min(best_cost_for_goal, cost)

            # Option B: Need to take a new image and then run bypa1zm7
            # For each rover that is eligible to mount/take, check if there exists a camera that:
            #  - can image the objective,
            #  - supports the required mode,
            #  - can be mounted on that rover.
            possible_cameras = set()
            for cam in self.camera_objectives.get(obj, ()):
                if mode in self.camera_modes.get(cam, ()):
                    # cam supports objective and requested mode
                    possible_cameras.add(cam)

            # For each eligible rover, and for each image waypoint for the objective, compute cost:
            # dist(current -> image_wp) + mount(1) + take(1) + dist(image_wp -> RL70M) + bypa1zm7(1)
            image_waypoints = self.wecz_obj_to_waypoints.get(obj, set())
            if possible_cameras and image_waypoints:
                # iterate rovers that are both in rover_positions and in rovers_with_mount
                for rover, cur_wp in rover_positions.items():
                    if rover not in self.rovers_with_mount:
                        continue
                    # check there is at least one camera from possible_cameras that can be mounted on this rover
                    cam_mountable = False
                    for cam in possible_cameras:
                        if rover in self.camera_rovers.get(cam, ()):
                            cam_mountable = True
                            break
                    if not cam_mountable:
                        continue

                    # BFS distances from rover's current location
                    dist_from_cur = self._bfs_distances(rover, cur_wp)

                    # For each image waypoint, compute distances
                    for img_wp in image_waypoints:
                        d1 = dist_from_cur.get(img_wp, float("inf"))
                        if d1 == float("inf"):
                            continue  # unreachable for this rover

                        # After taking image at img_wp, the rover must (maybe) move to RL70M.
                        dist_from_img = self._bfs_distances(rover, img_wp)
                        d2 = min((dist_from_img.get(rwp, float("inf")) for rwp in self.rl_candidates), default=float("inf"))

                        if d2 == float("inf"):
                            # No RL70M reachable after taking image -> maybe impossible from this rover/img_wp
                            continue

                        cost = d1 + 2 + d2 + 1  # move to image + mount + take + move to RL + bypa1zm7
                        best_cost_for_goal = min(best_cost_for_goal, cost)

            # As fallback, if nothing applicable was found, keep LARGE_COST; otherwise add best cost.
            total_cost += best_cost_for_goal

        # Ensure heuristic returns an integer finite value
        if total_cost >= LARGE_COST:
            # If at least one goal appears unreachable with our static analysis,
            # return a large finite number but not infinity.
            return LARGE_COST * len(self.goal_obj_mode_pairs)

        return int(total_cost)