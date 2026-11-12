from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens of a PDDL fact string without the surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic tailored to the secret_domain (du770mtw). The domain's goals require
    producing ne6sl557 facts for certain wqid8mog objects ("targets"). Producing
    a ne6sl557 for a target requires:
      - the target (wqid8mog) to be at some mz6r1d30 location L (this location is fixed;
        targets cannot be moved by available actions),
      - a pt4b08k4 object (call it "pt") to be at the same location L,
      - a fact (fqyrzl30 pt du) linking that pt to some du770mtw resource du,
      - the chosen du must currently be "available" (ndg963qz du).
    The domain provides two movement/processing actions affecting these facts:
      - nm6lzp2x: moves a pt along directed lpmls56r edges between mz6r1d30 nodes,
      - jql8txb3: creates fqyrzl30(pt,du) when pt and du co-locate at the same mz6r1d30,
      - f7y6nzq3: consumes of6wn8m4(target) and ndg963qz(du), producing ne6sl557(target)
        provided pt is at the same mz6r1d30 as the target and fqyrzl30(pt,du) holds.
    The heuristic estimates the minimal number of actions required to achieve all
    remaining ne6sl557 goals by reasoning about:
      - directed distances (in number of nm6lzp2x actions) between mz6r1d30 locations;
      - whether a suitable (pt, du) pair is already linked (fqyrzl30) and/or co-located;
      - the fact that each du (resource) can be used at most once (ndg963qz is deleted
        when the du is used).
    The heuristic is not required to be admissible; it aims to be informative and
    efficiently computable to guide greedy best-first search.

    Assumptions
    -----------
    - The planner's internal state represents facts as strings like "(predicate arg1 arg2)".
    - All target objects that must become ne6sl557 are listed in the task goals as
      "(ne6sl557 <obj>)".
    - Targets (wqid8mog) and du770mtw resources have immutable identities and their
      initial mz6r1d30 locations (qkdx0xe2 facts) are available in the state. Targets
      cannot be moved by available actions (only pt4b08k4 can be moved via nm6lzp2x).
    - lpmls56r static facts define directed edges between mz6r1d30 nodes; nm6lzp2x
      moves pt4b08k4 along these directed edges one step per action.
    - If a needed resource (du at a location where linking is possible) does not
      exist or a pt cannot reach the required locations via directed lpmls56r paths,
      the heuristic will return a large finite value (indicating difficulty/impossibility).

    Heuristic Initialization
    ------------------------
    In __init__ the heuristic:
      - Stores the set of goal facts (for quick goal checks).
      - Extracts the directed graph of mz6r1d30 nodes from static lpmls56r facts.
      - Prepares helper structures; distances are computed on demand (BFS) using this graph.
      - No expensive global search or grounding is performed at construction time.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a planner node (with node.state being a frozenset of fact strings):

    1. Quick goal check:
       - If all goal facts are already present in node.state, return 0.

    2. Extract current dynamic facts from the state:
       - qkdx0xe2 X L  -> maps object X (could be pt, du, or target) to mz6r1d30 L.
       - fqyrzl30 pt du -> set of existing links between pts and dus.
       - ndg963qz du -> set of available du resources (each du can be used once).
       - of6wn8m4 target -> target must have this precondition to be processed;
         if missing for a goal target, the target is considered impossible to process
         (we return a large finite cost).
       - The set of target objects is taken from the task goals (ne6sl557 facts).

    3. Identify candidate pts:
       - Any object that has a qkdx0xe2 location and is not classified as a du
         (has ndg963qz) or a target (appears in goals) is treated as a pt candidate.
       - Also any object that appears as the first argument of an fqyrzl30 fact is
         considered a pt candidate even if it is not classified elsewhere.

    4. Precompute directed shortest-path distances (BFS) over the static lpmls56r graph
       between relevant mz6r1d30 nodes (we compute distances from a pt's location and/or
       a du's location when needed). Distances represent the minimum number of nm6lzp2x
       actions required to move a pt from one mz6 node to another following direction.

    5. For each goal target that is not yet satisfied:
       - Let ris_loc be the target's fixed location from qkdx0xe2 facts. If missing, the
         target is unreachable -> large finite cost.
       - The target must be processed using some available du (ndg963qz). Each du can be
         used once; the heuristic enforces this by a greedy assignment (see below).
       - For each available du, compute the minimal action cost to process this target
         using any pt candidate:
           a) If fqyrzl30(pt,du) already exists:
                cost = dist(pt_loc -> ris_loc) + 1  # move pt to target (nm6lzp2x actions) + f7y6nzq3
           b) If no link exists:
                cost = dist(pt_loc -> du_loc) + 1   # move pt to du_loc + jql8txb3
                     + dist(du_loc -> ris_loc) + 1  # move pt from du_loc to ris_loc + f7y6nzq3
           (If any required directed path does not exist, that (pt,du) option is invalid.)
       - Record the cheapest (du, pt) option for the target.

    6. Assign dus to targets:
       - Using a greedy matching: sort unsatisfied targets by their cheapest option cost
         and, in that order, allocate the cheapest still-available du to each target.
       - If a target cannot be assigned any du (no remaining du options), the heuristic
         returns a large finite value.

    7. Sum the costs of the chosen (pt,du) plans for all targets and return the total.

    Notes on cost counts:
      - Each nm6lzp2x invocation counts as 1 action per edge moved.
      - jql8txb3 and f7y6nzq3 each count as a single action.
      - The heuristic is monotone in the sense that using a du consumes it (we model this),
        so the heuristic discourages plans that unrealistically reuse the same du repeatedly.
    """

    def __init__(self, task):
        # store goal facts (strings)
        self.goals = task.goals
        # extract static directed edges between mz6r1d30 nodes from lpmls56r facts
        self.adj = {}  # adjacency: node -> set(neighbor_nodes)
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "lpmls56r":
                # lpmls56r from -> to
                _, frm, to = parts
                self.adj.setdefault(frm, set()).add(to)
                # Ensure nodes exist in adjacency map even if no outgoing edges
                self.adj.setdefault(to, set())
        # cache for BFS distance computations: (source) -> {target: distance}
        self._distance_cache = {}

    def _bfs_distances(self, src):
        """
        Compute directed shortest-path distances from src to all reachable mz6 nodes.
        Results cached in self._distance_cache.
        """
        if src in self._distance_cache:
            return self._distance_cache[src]
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._distance_cache[src] = dist
        return dist

    def __call__(self, node):
        state = node.state

        # 1) Quick goal check: if all goals satisfied, return 0
        if self.goals <= state:
            return 0

        # Parse dynamic facts from state
        qkdx_locs = {}  # object -> mz6 node
        fq_links = set()  # (pt, du)
        ndg_available = set()  # set of du objects available (ndg963qz)
        of6_set = set()  # objects that have of6wn8m4 (must be true before processing)
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "qkdx0xe2":
                obj, loc = parts[1], parts[2]
                qkdx_locs[obj] = loc
            elif pred == "fqyrzl30":
                pt, du = parts[1], parts[2]
                fq_links.add((pt, du))
            elif pred == "ndg963qz":
                du = parts[1]
                ndg_available.add(du)
            elif pred == "of6wn8m4":
                of6_set.add(parts[1])

        # Determine goal target objects from self.goals (expected format: (ne6sl557 obj))
        target_objs = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "ne6sl557":
                target_objs.append(parts[1])

        # Identify pts: objects that appear as fqyrzl30 first arg or appear in qkdx0xe2 but not a du and not a target
        du_objects = set(ndg_available)  # known du objects (presently available)
        # Some dus might not be available (ndg not present) but could still exist; however
        # they are unusable if ndg is not present.
        pt_candidates = set()
        # First, any first arg of fqyrzl30 is a pt
        for pt, _ in fq_links:
            pt_candidates.add(pt)
        # Next, any object with a qkdx0xe2 location that is not identified as du or target is a pt candidate
        for obj, loc in qkdx_locs.items():
            if obj not in du_objects and obj not in target_objs:
                pt_candidates.add(obj)

        # Pre-checks and data structures for assignment
        # Map targets not yet satisfied
        unsatisfied_targets = []
        for t in target_objs:
            # If the goal already satisfied, skip
            if f"(ne6sl557 {t})" in state:
                continue
            unsatisfied_targets.append(t)

        # If no unsatisfied targets -> return 0 (caught earlier, but safe)
        if not unsatisfied_targets:
            return 0

        # Sanity large cost (finite but large)
        LARGE_COST = 100000

        # For each unsatisfied target, compute its location and verify it has of6wn8m4 (necessary precondition)
        target_location = {}
        for t in unsatisfied_targets:
            if t not in qkdx_locs:
                # target location unknown -> impossible to process
                return LARGE_COST
            if t not in of6_set:
                # target lacks of6wn8m4 precondition (cannot be processed)
                return LARGE_COST
            target_location[t] = qkdx_locs[t]

        # Build a local mutable availability set for du resources (we will greedily allocate)
        du_available_local = set(ndg_available)

        # For every du that exists in qkdx_locs but not in ndg_available, it cannot be used.
        # We'll need du locations for computing costs.
        du_locations = {du: qkdx_locs.get(du) for du in du_available_local}

        # Precompute candidate list of pts with their current locations
        pt_locations = {pt: qkdx_locs.get(pt) for pt in pt_candidates if pt in qkdx_locs}

        # If no pts available to use, impossible
        if not pt_locations:
            return LARGE_COST

        # For each unsatisfied target compute per-du minimal cost (over pts) and record best option
        target_options = {}  # target -> list of (cost, du, best_pt, details)
        for t in unsatisfied_targets:
            t_loc = target_location[t]
            options = []
            # For each available du try to compute minimal cost using any pt
            for du in du_available_local:
                du_loc = du_locations.get(du)
                # du_loc could be None (du exists but no qkdx0xe2 fact) - jql8txb3 requires du to be located to create link
                if du_loc is None:
                    # We cannot create a link with this du (no location), but we might already have an existing link
                    pass
                best_cost_for_du = None
                best_pt_for_du = None
                # Check all pt candidates
                for pt, pt_loc in pt_locations.items():
                    if pt_loc is None:
                        continue
                    # Case A: existing link (pt,du) present -> only need to move pt to target and then f7
                    if (pt, du) in fq_links:
                        # compute distance from pt_loc -> t_loc
                        dist_map = self._bfs_distances(pt_loc)
                        move_cost = dist_map.get(t_loc)
                        if move_cost is None:
                            continue
                        cost = move_cost + 1  # +1 for f7 action
                        if best_cost_for_du is None or cost < best_cost_for_du:
                            best_cost_for_du = cost
                            best_pt_for_du = pt
                    else:
                        # Need to create link first: move pt to du_loc (nm6lzp2x steps), jql, move pt to target, f7
                        if du_loc is None:
                            # cannot create link with du that has no known location
                            continue
                        dist_map_to_du = self._bfs_distances(pt_loc)
                        d1 = dist_map_to_du.get(du_loc)
                        if d1 is None:
                            continue
                        dist_map_du = self._bfs_distances(du_loc)
                        d2 = dist_map_du.get(t_loc)
                        if d2 is None:
                            continue
                        cost = d1 + 1 + d2 + 1  # move to du + jql + move to target + f7
                        if best_cost_for_du is None or cost < best_cost_for_du:
                            best_cost_for_du = cost
                            best_pt_for_du = pt
                if best_cost_for_du is not None:
                    options.append((best_cost_for_du, du, best_pt_for_du))
            if not options:
                # No available du can be used to process this target
                return LARGE_COST
            # sort options by cost ascending for this target (we'll pick greedily later)
            options.sort(key=lambda x: x[0])
            target_options[t] = options

        # Greedy assignment of du to targets:
        # Sort targets by their minimal possible cost (so cheap-to-serve targets get allocated first).
        targets_sorted = sorted(unsatisfied_targets, key=lambda t: target_options[t][0][0])

        total_cost = 0
        used_du = set()
        for t in targets_sorted:
            assigned = False
            # try the cheapest options in order
            for cost, du, pt in target_options[t]:
                if du in used_du:
                    continue
                # assign this du to target
                used_du.add(du)
                total_cost += cost
                assigned = True
                break
            if not assigned:
                # no du remained available for this target
                return LARGE_COST

        return total_cost