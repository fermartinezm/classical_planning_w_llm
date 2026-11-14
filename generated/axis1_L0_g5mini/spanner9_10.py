from collections import deque
from heuristics.heuristic_base import Heuristic

INF = 10 ** 6


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact string into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'spanner' domain.

    The domain requires a man to pick up usable spanners and walk to nuts to tighten them.
    Each tighten action consumes the spanner (makes it not usable). Walk actions move the
    man along directed 'link' edges. Picking up a spanner and tightening a nut each cost
    one action. The heuristic estimates the remaining number of actions (walks + pickups +
    tightens) required to turn all currently loose nuts into tightened ones.

    Assumptions
    -----------
    - There is typically a single man, but the code locates any object that behaves like a man
      (appears in 'carrying' or is an 'at' fact and is not a spanner/nut).
    - Walk actions follow directed 'link' facts from static information. Links are treated as
      directed edges.
    - A spanner can be used once if it is marked 'usable'. After a tighten, the spanner becomes
      not usable.
    - The man can carry multiple spanners simultaneously (the domain permits multiple (carrying)
      facts).
    - If there are fewer usable spanners available than loose nuts, the state is treated as
      effectively unsolvable in the short term and the heuristic returns a large finite value.
      (Requirement is only that the heuristic be finite for solvable states.)

    Heuristic Initialization
    ------------------------
    - Extract directed adjacency information from static 'link' facts and store it as a graph.
    - Store the set of goal facts (used to return 0 when already at goal).
    - No other persistent static data is required.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    The heuristic greedily simulates a plausible sequence of actions to tighten all loose nuts:
      1. Parse the current state to find:
         - positions of objects given by 'at' facts,
         - which spanners are usable and whether they are currently carried,
         - which nuts are currently loose and their locations,
         - the man's current location.
      2. Precompute distances (minimum number of directed 'walk' actions) between locations
         using BFS on the directed link graph. Distances are cached per source location.
      3. If all goal facts already hold in the state, return 0.
      4. If the number of usable spanners is less than the number of loose nuts, return a large
         finite cost (heuristic is not required to be correct for unsolvable states).
      5. Otherwise, repeatedly choose the next tightening to perform greedily:
         - If the man currently carries at least one usable spanner, pick the closest loose nut
           (in walk-distance) and pay the cost to walk there plus 1 for the tighten action.
           Consume one carried usable spanner and remove the tightened nut from consideration.
         - If the man carries no usable spanner, choose a usable (not-yet-used) spanner and a nut
           pair that minimizes:
               cost = dist(man_loc, spanner_loc) + 1 (pickup) + dist(spanner_loc, nut_loc) + 1 (tighten)
           and execute that pair (move man to nut location and remove that spanner and nut).
      6. Sum all action costs encountered in this greedy simulation and return that sum as the
         heuristic estimate.
      Notes:
        - The heuristic is not admissible, because it ignores some opportunities for reusing
          travel (e.g., picking multiple spanners along a route) and double-counts travel costs
          across independent nut-handling choices. This is acceptable for guiding greedy
          best-first search where accuracy and pruning are more important than admissibility.
        - Distances that are unreachable lead to a large heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize heuristic using static facts from the task.

        - task.goals: set of goal fact strings
        - task.static: set of static fact strings (used to build the directed graph)
        """
        self.goals = task.goals
        self.static = task.static

        # Build directed adjacency list from 'link' facts in static info.
        self.adj = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "link" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, []).append(b)

        # Cache of BFS distance maps keyed by source location
        self._dist_cache = {}

    def _dist_from(self, source):
        """Return a dict mapping locations to distance (number of walk actions) from source."""
        if source in self._dist_cache:
            return self._dist_cache[source]

        distances = {source: 0}
        q = deque([source])
        while q:
            cur = q.popleft()
            for nbr in self.adj.get(cur, []):
                if nbr not in distances:
                    distances[nbr] = distances[cur] + 1
                    q.append(nbr)
        self._dist_cache[source] = distances
        return distances

    def _dist(self, a, b):
        """Return distance (walk actions) from location a to b; INF if unreachable."""
        if a == b:
            return 0
        distances = self._dist_from(a)
        return distances.get(b, INF)

    def __call__(self, node):
        """
        Compute the heuristic for the given search node.

        Returns:
            An integer estimate of the number of actions to reach a goal.
        """
        state = node.state

        # 0 for goal states (check first per checklist)
        if self.goals <= state:
            return 0

        # Parse state facts: build maps for 'at', 'carrying', 'usable', 'loose', 'tightened'
        at_map = {}  # obj -> location
        carrying = []  # list of (man, spanner)
        usable_spanners = set()
        loose_nuts = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                at_map[obj] = loc
            elif pred == "carrying" and len(parts) >= 3:
                man, sp = parts[1], parts[2]
                carrying.append((man, sp))
            elif pred == "usable" and len(parts) >= 2:
                usable_spanners.add(parts[1])
            elif pred == "loose" and len(parts) >= 2:
                loose_nuts.add(parts[1])

        # Determine set of spanner names and nut names appearing in the state
        spanner_names = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] in ("usable",):
                spanner_names.add(parts[1])
        # Also include spanners that appear in at or carrying but may not be 'usable'
        for obj, loc in at_map.items():
            # If an object is referenced by 'usable' it is already included. We don't know types
            # for others, but we only need spanners for matching usable ones; still we want to
            # collect potential spanner names that are being carried.
            pass

        # Identify the man name. Prefer the first carrying fact's man; otherwise choose any
        # object that appears in an at fact and is not a nut or a spanner (we detect spanners via usable set).
        man_name = None
        if carrying:
            man_name = carrying[0][0]
        else:
            # candidates are at_map keys that are not obviously spanners or nuts
            for obj in at_map:
                if obj not in loose_nuts and obj not in usable_spanners:
                    man_name = obj
                    break
            # fallback: if still None pick any at_map key (should not happen in correct tasks)
            if man_name is None and at_map:
                man_name = next(iter(at_map.keys()))

        # Build spanners dictionary with usable & carried info and location (if not carried)
        spanners = {}
        # Initialize from usable set (these are the only spanners that can help)
        for s in usable_spanners:
            spanners[s] = {"usable": True, "carried": False, "loc": at_map.get(s, None)}
        # Mark carried spanners as carried (they might or might not be in usable_spanners)
        for man, s in carrying:
            if s not in spanners:
                # if it's carried but not marked usable, it cannot be used for tightening; ignore
                spanners[s] = {"usable": False, "carried": True, "loc": None}
            else:
                spanners[s]["carried"] = True
                spanners[s]["loc"] = None  # carried -> location is currently the man

        # Filter to only usable spanners (we only can use those)
        usable_spanners_info = {
            s: info for s, info in spanners.items() if info["usable"]
        }

        # If there are no loose nuts, cost is 0
        if not loose_nuts:
            return 0

        # If not enough usable spanners to tighten all loose nuts, return a large finite cost
        if len(usable_spanners_info) < len(loose_nuts):
            return 1000 + 10 * (len(loose_nuts) - len(usable_spanners_info))

        # Determine man's current location
        man_loc = at_map.get(man_name, None)
        if man_loc is None:
            # If man has no at fact, we cannot reason; return large finite estimate
            return 10000

        # Prepare mutable structures for greedy simulation
        remaining_nuts = {n: at_map.get(n, None) for n in loose_nuts}
        # If any nut has unknown location, treat as unreachable -> large estimate
        for n, loc in remaining_nuts.items():
            if loc is None:
                return 10000

        # copy usable spanners info into a mutable dict keyed by spanner name
        available_spanners = {
            s: {"carried": info["carried"], "loc": info["loc"]} for s, info in usable_spanners_info.items()
        }

        total_cost = 0
        cur_loc = man_loc

        # Greedy loop: at each step, if any carried usable spanners exist, use one on nearest nut;
        # otherwise select the best (spanner, nut) pair minimizing travel+pickup+travel+tighten costs.
        while remaining_nuts:
            # Count available carried usable spanners
            carried_list = [s for s, info in available_spanners.items() if info["carried"]]
            if carried_list:
                # choose nearest nut from cur_loc
                best_nut = None
                best_dist = INF
                for nut, nut_loc in remaining_nuts.items():
                    d = self._dist(cur_loc, nut_loc)
                    if d < best_dist:
                        best_dist = d
                        best_nut = nut
                if best_dist >= INF:
                    return 10000
                # walk to nut (best_dist) and tighten (1)
                total_cost += best_dist + 1
                # consume one carried spanner (remove an arbitrary one from carried_list)
                used_sp = carried_list[0]
                del available_spanners[used_sp]
                # remove nut
                del remaining_nuts[best_nut]
                # update man's location
                cur_loc = at_map.get(best_nut, cur_loc)
            else:
                # need to pick up a spanner first. Choose best spanner-nut pair.
                best_pair = None
                best_pair_cost = INF
                for s, info in available_spanners.items():
                    s_loc = info["loc"]
                    if s_loc is None:
                        # spanner not carried but has no known location; skip
                        continue
                    d1 = self._dist(cur_loc, s_loc)  # man to spanner
                    if d1 >= INF:
                        continue
                    for nut, nut_loc in remaining_nuts.items():
                        d2 = self._dist(s_loc, nut_loc)  # spanner location to nut
                        if d2 >= INF:
                            continue
                        cost = d1 + 1 + d2 + 1  # walk to spanner + pickup + walk to nut + tighten
                        if cost < best_pair_cost:
                            best_pair_cost = cost
                            best_pair = (s, nut, s_loc, nut_loc)
                if best_pair is None:
                    # no feasible pair (unreachable) -> large estimate
                    return 10000
                s, nut, s_loc, nut_loc = best_pair
                total_cost += best_pair_cost
                # remove that spanner and nut from availability
                del available_spanners[s]
                del remaining_nuts[nut]
                # update man location to nut location
                cur_loc = nut_loc

        return int(total_cost)