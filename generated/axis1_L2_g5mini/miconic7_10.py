from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Remove surrounding parentheses and split the fact into parts.
    Example: "(pvhp8m0p a b)" -> ["pvhp8m0p", "a", "b"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (unkhrp2l) domain.

    The domain has:
    - A single movable marker represented by facts of the form (v0yy74n7 <loc>).
    - Directed adjacency facts (pvhp8m0p <from> <to>) describing where the marker can move in one action.
    - Objects of type jxhpkqr5 that can be transformed into the goal predicate (guby8wxu <obj>)
      through two domain actions:
        * makpz2tc: when the marker is at the location where the object currently "is"
                   (pxch7pi8 <obj> <loc_px>) it converts pxch7pi8 -> eacj8ftl.
        * jg64qouf: when the marker is at the object's target location (pahbmn6t <obj> <loc_pah>)
                   and the object has eacj8ftl it converts eacj8ftl -> guby8wxu.
    - Thus, to make an object reach guby8wxu, the marker typically must visit the object's px location
      (to create eacj8ftl) and then the object's pah location (to finish the conversion).
    The heuristic estimates the number of actions (marker moves and conversion actions) required to
    satisfy all guby8wxu goals from the current state by simulating a greedy sequence of visits
    to the required locations.

    Assumptions
    -----------
    - Each application of jt9wnwgs or v90zoaro (marker move) counts as 1 action and corresponds to
      traversing one directed pvhp8m0p edge.
    - makpz2tc and jg64qouf each count as 1 action.
    - pahbmn6t facts (mapping object -> target location) and the pvhp8m0p edges are static and available
      in the Task.static set.
    - pxch7pi8 facts (object current locations) and eacj8ftl facts are part of the dynamic state.
    - If facts needed for an object (pxch7pi8 or pahbmn6t) are missing or unreachable according to the
      static graph, the heuristic returns a large finite value (treating the object as currently unreachable).
    - Heuristic is not required to be admissible; it aims to provide a good guided estimate for greedy best-first search.

    Heuristic Initialization
    ------------------------
    - Store the goal facts (task.goals).
    - Extract the directed adjacency graph from static pvhp8m0p facts.
    - Extract the mapping from object -> target location from static pahbmn6t facts.
    - Prepare an adjacency dict for fast BFS distance computations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state satisfies all goals, return 0.
    2. Identify the marker's current location by finding the single fact of the form (v0yy74n7 <loc>).
    3. For each goal object (from the goals set with predicate guby8wxu):
       - If the object already has (guby8wxu <obj>) in the state, it requires 0 actions.
       - Otherwise:
         a) Check if (eacj8ftl <obj>) is already true in the state:
            - If yes: the only remaining step is to have the marker visit the object's pah location
              (from static pahbmn6t) and apply jg64qouf: cost = distance(current_marker, loc_pah) + 1.
         b) If no: the object normally needs two actions:
            - Visit object's px location (found from state as pxch7pi8 <obj> <loc_px>),
              perform makpz2tc (1 action) to get eacj8ftl, then move to the object's pah location
              and perform jg64qouf (1 action).
            - The cost is therefore: distance(current_marker, loc_px) + 1 + distance(loc_px, loc_pah) + 1.
       - If any required location is missing or unreachable in the directed graph, treat the object as unreachable
         and return a large finite heuristic.
    4. To avoid gross overestimation by assuming the marker teleports back to the start for each object,
       we simulate the marker performing conversions sequentially in a greedy order:
       - Start at the marker's current location.
       - Repeatedly select the remaining object whose next required visit (loc_px if makpz needed else loc_pah)
         is closest (in terms of directed shortest-path distance) to the marker's current location.
       - Add the travel cost and the required conversion actions for that object, update the current marker
         location to the object's pah location (where conversion finishes), and remove that object from the
         remaining set.
    5. Sum the costs of this greedy sequence and return it as the heuristic estimate.
    """

    def __init__(self, task):
        # Goals are given as strings like "(guby8wxu obj)"
        self.goals = set(task.goals)
        self.static = set(task.static)

        # Build directed adjacency graph from pvhp8m0p facts in static information.
        # pvhp8m0p <from> <to> means one marker move unit from <from> to <to>.
        self.adj = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "pvhp8m0p" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.adj.setdefault(src, []).append(dst)
                # Note: keep edges directed; there are separate domain actions for reverse moves.

        # Build object -> pah location mapping from static pahbmn6t facts.
        # We store only one target location per object here (domain instances use single mapping).
        self.pah_map = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "pahbmn6t" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                # If multiple, last one will remain; typical instances have one mapping.
                self.pah_map[obj] = loc

        # Cache for BFS distances per source node to speed repeated distance queries.
        self._bfs_cache = {}

        # A large finite value used when something appears unreachable.
        self._INF = 10 ** 6

    def _bfs_distances(self, src):
        """
        Compute shortest directed distances (in number of edges) from src to all reachable nodes
        using BFS. Cache results for reuse.
        """
        if src in self._bfs_cache:
            return self._bfs_cache[src]

        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._bfs_cache[src] = dist
        return dist

    def _distance(self, src, dst):
        """
        Return shortest directed distance from src to dst (number of marker moves).
        If dst not reachable, return None.
        """
        if src == dst:
            return 0
        dist_map = self._bfs_distances(src)
        return dist_map.get(dst, None)

    def __call__(self, node):
        """
        Compute the heuristic value for the given search node.

        Returns:
            An integer estimate of remaining actions (marker moves + makpz2tc + jg64qouf) required
            to make all goal (guby8wxu ...) facts true.
        """
        state = node.state

        # 1) If already goal state, heuristic is 0 (and only then we return 0).
        # The Task.goal_reached is expected to be used externally; we replicate the check here:
        # If all goal facts are present in the state, return 0.
        if self.goals <= state:
            return 0

        # 2) Find current marker location: fact of form (v0yy74n7 <loc>)
        marker_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "v0yy74n7" and len(parts) >= 2:
                marker_loc = parts[1]
                break
        if marker_loc is None:
            # No marker found in state -> treat as unreachable but finite.
            return self._INF

        # 3) Determine goal objects (from self.goals) and prepare per-object info.
        # Extract list of objects that must become guby8wxu (goal facts).
        goal_objects = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "guby8wxu" and len(parts) >= 2:
                goal_objects.append(parts[1])

        # For each object compute:
        # - whether already achieved (skip),
        # - whether eacj8ftl present (so only jg64qouf needed),
        # - current px location (from state via pxch7pi8),
        # - pah location (from static pah_map).
        objs_info = {}
        # Pre-scan pxch7pi8 facts in state for object current locations.
        px_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "pxch7pi8" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                px_locations[obj] = loc

        # Pre-scan eacj8ftl facts in state.
        eac_present = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "eacj8ftl" and len(parts) >= 2:
                eac_present.add(parts[1])

        # Build objects info
        for obj in goal_objects:
            if f"(guby8wxu {obj})" in state:
                # already satisfied
                continue
            info = {}
            info["eac"] = obj in eac_present
            info["loc_px"] = px_locations.get(obj, None)
            info["loc_pah"] = self.pah_map.get(obj, None)
            objs_info[obj] = info

        # If nothing to achieve (all goals satisfied), already returned 0 earlier.
        if not objs_info:
            return 0

        # 4) Greedy simulation: repeatedly pick nearest next-required-site object to current marker location.
        remaining = set(objs_info.keys())
        current = marker_loc
        total_cost = 0

        # Clear BFS cache to avoid stale info across calls.
        # (However, keep adjacency static; cache keyed by node name.)
        self._bfs_cache.clear()

        while remaining:
            best_obj = None
            best_cost_travel = None
            best_dist1 = None
            best_dist_between = None
            best_first = None
            best_second = None

            # Evaluate each remaining object's immediate travel needs from current location.
            for obj in remaining:
                info = objs_info[obj]
                loc_pah = info["loc_pah"]
                # If object has eac, only need to visit loc_pah and perform jg64qouf.
                if info["eac"]:
                    if loc_pah is None:
                        # missing mapping -> unreachable
                        return self._INF
                    d1 = self._distance(current, loc_pah)
                    if d1 is None:
                        return self._INF
                    # total travel to finish this object starting from current (no intermediate)
                    travel = d1
                    # choose minimal travel
                    if best_obj is None or travel < best_cost_travel:
                        best_obj = obj
                        best_cost_travel = travel
                        best_dist1 = d1
                        best_dist_between = 0
                        best_first = loc_pah
                        best_second = None
                else:
                    # Need makpz2tc at loc_px then jg64qouf at loc_pah
                    loc_px = info["loc_px"]
                    if loc_px is None or loc_pah is None:
                        # missing necessary locations -> unreachable
                        return self._INF
                    d1 = self._distance(current, loc_px)
                    if d1 is None:
                        return self._INF
                    d_between = self._distance(loc_px, loc_pah)
                    if d_between is None:
                        return self._INF
                    travel = d1 + d_between
                    if best_obj is None or travel < best_cost_travel:
                        best_obj = obj
                        best_cost_travel = travel
                        best_dist1 = d1
                        best_dist_between = d_between
                        best_first = loc_px
                        best_second = loc_pah

            # If for some reason we didn't pick an object, break to avoid infinite loop.
            if best_obj is None:
                return self._INF

            # Compute cost contribution for chosen object.
            info = objs_info[best_obj]
            if info["eac"]:
                # travel to loc_pah + jg64qouf (1)
                total_cost += best_dist1 + 1
                current = best_first  # after jg64qouf marker remains at loc_pah
            else:
                # travel to loc_px + makpz2tc (1) + travel loc_px->loc_pah + jg64qouf (1)
                total_cost += best_dist1 + 1 + best_dist_between + 1
                current = best_second  # end at loc_pah

            remaining.remove(best_obj)

            # Safety cap: if cost grows beyond INF, return INF
            if total_cost >= self._INF:
                return self._INF

        # Return the greedy-sequence total cost as heuristic estimate.
        return int(total_cost)