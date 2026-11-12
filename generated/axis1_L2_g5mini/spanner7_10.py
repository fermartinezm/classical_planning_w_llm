from collections import deque
from heuristics.heuristic_base import Heuristic

INF = 10 ** 6


def get_parts(fact):
    """Return list of tokens of a PDDL fact string (without surrounding parentheses)."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (du770mtw) domain.

    This heuristic estimates the number of actions required to achieve all
    goal facts of the form (ne6sl557 <wqid8mog>) from a given state. The
    domain's high-level structure is:
      - Objects of type wqid8mog (call them "resources" or "requests") have a
        location label via (qkdx0xe2 <obj> <mz-node>) and may be consumable
        via (of6wn8m4 ...). The goals ask for (ne6sl557 <wqid8mog>).
      - There are pt4b08k4 objects (call them "tools" or "tokens") that also
        have (qkdx0xe2 <pt> <mz-node>) and that can be linked to du770mtw
        objects (call them "stations") via (fqyrzl30 <pt> <station>).
      - Stations (du770mtw) may be available as resources (ndg963qz).
      - Movement of a pt's qkdx0xe2 label along mz-nodes is possible via
        nm6lzp2x when a static edge (lpmls56r <from> <to>) exists. Each such
        movement corresponds to one action.
      - A linking action jql8txb3 requires both a station and a pt to be at
        the same mz-node (same qkdx0xe2 value) and produces (fqyrzl30 <pt>
        <station>) (cost 1).
      - The delivering action f7y6nzq3 requires:
          * the station to be available (ndg963qz <station>),
          * the pt to be linked to that station (fqyrzl30 <pt> <station>),
          * the pt to be at the same mz-node as the target wqid8mog
            (qkdx0xe2 <pt> <mz> and qkdx0xe2 <wqid8mog> <mz>),
            and (of6wn8m4 <wqid8mog>) must hold.
        It produces (ne6sl557 <wqid8mog>) and consumes the station availability
        and the of6wn8m4 flag. The delivering action itself costs 1.
      - Therefore, to serve one goal <wqid8mog> the planner typically needs:
          * to choose an available station,
          * to (possibly) move some pt to the station's mz-node,
          * to link the pt to the station (jql8txb3) (unless already linked),
          * (possibly) move the linked pt to the wqid8mog's mz-node,
          * apply f7y6nzq3 to deliver.
        Movement steps correspond to nm6lzp2x actions (each edge one action).
        jql8txb3 and f7y6nzq3 each cost 1.

    Assumptions
    -----------
    - Static facts lpmls56r form a directed graph over mz6 nodes. Movement uses
      those directed edges.
    - qkdx0xe2 facts in the current state indicate the current mz-node for
      objects that can be moved (pts and wqid8mog and stations if present).
    - of6wn8m4 facts are required for a wqid8mog to be delivered; the heuristic
      assumes these are provided in the instance for goals that are meant to
      be achievable. If not present in the current state the heuristic assigns
      a large penalty.
    - Stations (ndg963qz) are single-use resources (they are consumed by
      f7y6nzq3); the heuristic models that by disallowing reuse of the same
      station for multiple goals (it uses a greedy assignment).
    - We ignore temporal interference between different pts (we allow reuse of
      pts in estimating costs) because modeling exact reuse leads to high
      computational cost; the heuristic therefore can be non-admissible but is
      efficient to compute.
    - If a pt is already linked to a station (fqyrzl30 present in state),
      the linking action (jql8txb3) is not required for that pt-station pair.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
      - self.goals: set of goal facts (so we can immediately return 0 on goal).
      - static adjacency for mz-nodes from lpmls56r facts (directed edges).
      - precompute BFS-based shortest path distances between all mz-nodes
        mentioned in the static graph or in dynamic qkdx0xe2 facts. Distances
        correspond to the minimum number of nm6lzp2x actions needed to move a
        pt from one mz-node to another following the directed edges.

    Step-By-Step Thinking for Computing Heuristic (for a given node/state)
    --------------------------------------------------------------------
    1. If all goals are satisfied in the current state, return 0.
    2. Parse the current state to obtain mappings:
         - qk_map[obj] = mz_node for every (qkdx0xe2 obj mz_node) fact.
         - fq_map[pt] = set of stations for every (fqyrzl30 pt station) fact.
         - available_stations = set of station objects with (ndg963qz station).
         - available_of = set of wqid8mog with (of6wn8m4 ...) in state.
         - already_done = set of wqid8mog with (ne6sl557 ...) in state.
    3. Identify the set of goal wqid8mog objects (goals that still need
       achieving = goals \ already_done).
    4. Identify pt candidates as objects that have a qkdx0xe2 mapping and do
       not appear to be stations or wqid8mog (approximated by removing
       available_stations and objects present in available_of or already_done).
       This heuristic uses this simple typing inference to cheaply find pts.
    5. For each remaining goal w:
         - If w has no qkdx0xe2 mapping in the current state, the goal looks
           unreachable; assign a large penalty for that goal.
         - Otherwise, let target_node = qk_map[w].
         - For each available station s (with qkdx0xe2 mapping station_node)
           and for each pt candidate p:
             * If p is already linked to s (fqyrzl30 p s in state):
                 cost(p,s) = distance(p_node -> target_node) + 1
                 (only movement to target_node if needed, plus f7 action).
             * Else:
                 cost(p,s) = distance(p_node -> station_node)
                              + 1  (jql8txb3 linking)
                              + distance(station_node -> target_node)
                              + 1  (f7 delivery)
               If any distance is unreachable (no directed path), treat as INF.
         - The best cost to serve w is the minimum cost across all (p,s) pairs.
    6. Because each station is single-use, compute a greedy assignment of
       stations to unsatisfied goals: sort goals by their best single-station
       cost and assign the cheapest available station to each goal in order,
       removing assigned stations from the pool. Sum assigned costs.
       (This is a fast approximate solver for the underlying assignment
       problem; it is not guaranteed optimal but is effective and cheap.)
    7. If some goal cannot be assigned a finite-cost station, add a large
       penalty for that goal. Return the sum of the computed costs as the
       heuristic estimate.

    Complexity / Performance Notes
    ------------------------------
    - All-pairs shortest paths on the (usually small) mz-node graph is done
      with BFS from each node (linear per source in number of edges).
    - The main loop then evaluates costs for each goal across stations and
      pts; this is polynomial in numbers of goals * stations * pts.
    - The heuristic is deliberately optimistic about pt re-use and ignores
      inter-goal sequencing beyond single-use stations; this makes it fast and
      effective for guiding greedy best-first search in typical instances.
    """

    def __init__(self, task):
        # Store goals as a set of fact strings.
        self.goals = set(task.goals)
        # Static facts (lpmls56r edges) are used to construct the movement graph.
        self.static = set(task.static)

        # Build directed adjacency for mz-nodes from static lpmls56r facts.
        self.adj = {}
        self._all_mz_nodes = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "lpmls56r" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.adj.setdefault(src, []).append(dst)
                self._all_mz_nodes.add(src)
                self._all_mz_nodes.add(dst)

        # We'll lazily complete the distance table once we see the first state.
        self._distances = None

    def _compute_all_pairs_distances(self, extra_nodes):
        """
        Compute directed shortest-path distances (in number of nm6lzp2x moves)
        between all mz-nodes discovered in static facts and in the current
        state's qkdx0xe2 facts.
        """
        nodes = set(self._all_mz_nodes) | set(extra_nodes)
        # Ensure nodes appear in adjacency map (even if no outgoing edges).
        for n in nodes:
            self.adj.setdefault(n, [])
        distances = {n: {} for n in nodes}
        for src in nodes:
            # BFS from src
            d = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in self.adj.get(u, []):
                    if v not in d:
                        d[v] = d[u] + 1
                        q.append(v)
            distances[src] = d
        self._distances = distances

    def __call__(self, node):
        state = node.state

        # If goal already reached, return 0.
        if self.goals <= state:
            return 0

        # Parse dynamic facts in the current state.
        qk_map = {}  # object -> mz-node from qkdx0xe2
        fq_map = {}  # pt -> set of stations from fqyrzl30
        available_stations = set()  # stations with ndg963qz
        available_of = set()  # wqid8mog objects with of6wn8m4 (required for delivery)
        already_done = set()  # wqid8mog objects with ne6sl557 (already satisfied)

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "qkdx0xe2" and len(parts) >= 3:
                obj, mz = parts[1], parts[2]
                qk_map[obj] = mz
            elif parts[0] == "fqyrzl30" and len(parts) >= 3:
                pt, station = parts[1], parts[2]
                fq_map.setdefault(pt, set()).add(station)
            elif parts[0] == "ndg963qz" and len(parts) >= 2:
                available_stations.add(parts[1])
            elif parts[0] == "of6wn8m4" and len(parts) >= 2:
                available_of.add(parts[1])
            elif parts[0] == "ne6sl557" and len(parts) >= 2:
                already_done.add(parts[1])

        # Identify goal wqid8mog objects that still need to be produced.
        goal_wids = []
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "ne6sl557" and len(parts) >= 2:
                wid = parts[1]
                if wid not in already_done:
                    goal_wids.append(wid)

        # Build set of mz-nodes that appear dynamically (for BFS computation).
        dynamic_mz_nodes = set(qk_map.values())

        # Compute all-pairs distances over mz-node graph, including dynamic nodes.
        if self._distances is None:
            self._compute_all_pairs_distances(dynamic_mz_nodes)
        else:
            # If new nodes appeared, recompute to include them.
            unknown_nodes = dynamic_mz_nodes - set(self._distances.keys())
            if unknown_nodes:
                self._compute_all_pairs_distances(dynamic_mz_nodes)

        # Infer candidate pts: objects with a qkdx0xe2 mapping that are not
        # identified as stations or wqid8mog (approximate typing).
        ris_like = set()
        # Collect objects that appear in of6wn8m4 or ne6sl557 (wqid8mog).
        ris_like.update(available_of)
        ris_like.update(already_done)
        candidate_pts = [o for o in qk_map.keys() if o not in available_stations and o not in ris_like]

        # For each goal, compute cheapest cost across (pt, station) pairs.
        # We'll then greedily assign stations to goals (single-use stations).
        # Precompute list of available stations that have a qkdx0xe2 mapping
        # (we need their mz-node for linking).
        stations_with_node = [s for s in available_stations if s in qk_map]

        # Pre-calc mapping from object -> mz node for quick access.
        def dist(a, b):
            """Directed distance from mz-node a to b (number of nm6lzp2x moves)."""
            if a is None or b is None:
                return INF
            if a == b:
                return 0
            dmap = self._distances.get(a)
            if not dmap:
                return INF
            return dmap.get(b, INF)

        # For each goal compute best costs for each station (min over pts).
        goal_best_costs = {}  # goal -> list of (cost, station, best_pt)
        for wid in goal_wids:
            if wid not in qk_map:
                # No location information for this wqid8mog: mark as hard.
                goal_best_costs[wid] = [(INF, None, None)]
                continue
            target_node = qk_map[wid]
            # If the required 'of6wn8m4' predicate is missing, f7 cannot be applied.
            if wid not in available_of:
                # very large penalty (goal currently not deliverable)
                goal_best_costs[wid] = [(INF, None, None)]
                continue

            per_station_candidates = []
            for s in stations_with_node:
                s_node = qk_map.get(s)
                best_cost_for_station = INF
                best_pt = None
                for p in candidate_pts:
                    p_node = qk_map.get(p)
                    if p_node is None:
                        continue
                    # If p is already linked to s, linking action not required.
                    already_linked = p in fq_map and s in fq_map[p]
                    if already_linked:
                        # Need to move p to target_node (if necessary) then apply f7.
                        move_cost = dist(p_node, target_node)
                        if move_cost >= INF:
                            cost_val = INF
                        else:
                            cost_val = move_cost + 1  # +1 for f7
                    else:
                        # Must move p to station node to link (dist p->s), link (1),
                        # then move station node -> target_node (dist), then f7 (1).
                        d1 = dist(p_node, s_node)
                        d2 = dist(s_node, target_node)
                        if d1 >= INF or d2 >= INF:
                            cost_val = INF
                        else:
                            cost_val = d1 + 1 + d2 + 1
                    if cost_val < best_cost_for_station:
                        best_cost_for_station = cost_val
                        best_pt = p
                per_station_candidates.append((best_cost_for_station, s, best_pt))
            if not per_station_candidates:
                # No usable stations with node info => unreachable for now.
                goal_best_costs[wid] = [(INF, None, None)]
            else:
                # Keep only finite candidates and sort by cost.
                per_station_candidates.sort(key=lambda x: x[0])
                goal_best_costs[wid] = per_station_candidates

        # Greedy assignment: pick cheapest goal-station pair iteratively.
        total = 0
        available_stations_set = set(stations_with_node)
        # Build list of goals with minimal cost for sorting
        goal_min_list = []
        for wid, cand_list in goal_best_costs.items():
            min_cost = cand_list[0][0] if cand_list else INF
            goal_min_list.append((min_cost, wid))
        goal_min_list.sort()

        for _, wid in goal_min_list:
            # For this goal, find the cheapest station that is still available.
            assigned = False
            for cost_val, s, best_pt in goal_best_costs.get(wid, []):
                if cost_val >= INF:
                    continue
                if s is None:
                    continue
                if s in available_stations_set:
                    total += cost_val
                    available_stations_set.remove(s)
                    assigned = True
                    break
            if not assigned:
                # If not assigned but there exist candidates with INF, add large penalty
                # to keep heuristic finite (but large).
                total += 1000

        return total if total < INF else INF