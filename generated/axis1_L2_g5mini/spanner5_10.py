from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (du770mtw). The domain's
    goals are facts of the form (ne6sl557 ?ris) where ?ris is an object of
    type wqid8mog. To achieve (ne6sl557 ?ris) the action f7y6nzq3 must be
    applied (it directly adds ne6sl557). The preconditions for f7y6nzq3
    require:
      - (of6wn8m4 ?ris)
      - (qkdx0xe2 ?ris ?ts)                     -- ris must be associated to some mz node ?ts
      - (fqyrzl30 ?n54xx ?cz)                    -- some pt4b08k4 ?n54xx linked to du770mtw ?cz
      - (qkdx0xe2 ?n54xx ?ts)                    -- the n54xx must be at the same mz node ?ts
      - (ndg963qz ?cz)                           -- the chosen cz must be "available"

    The domain provides two other actions that affect these facts:
      - jql8txb3 creates (fqyrzl30 ?n54xx ?cz) if both ?n54xx and ?cz are
        already associated with the same mz node ?ts (i.e., both have
        (qkdx0xe2 ... ?ts)). jql8txb3 deletes (qkdx0xe2 ?cz ?ts) but f7y6nzq3
        does not require (qkdx0xe2 ?cz ?ts), so that deletion is acceptable.
      - nm6lzp2x moves a pt4b08k4 object ?n54xx along a directed lpmls56r
        edge between mz nodes. Each nm6lzp2x application advances ?n54xx
        one step along the lpmls56r directed graph:
          precondition (qkdx0xe2 ?n54xx ?f) and (lpmls56r ?f ?w)
          effect moves (qkdx0xe2 ?n54xx ?f) -> (qkdx0xe2 ?n54xx ?w)

    Assumptions
    -----------
    - We assume action costs are uniform: each operator (nm6lzp2x, jql8txb3,
      f7y6nzq3) counts as one step.
    - A du770mtw object (cz) that has (ndg963qz ?cz) can be used at most
      once in practice because f7y6nzq3 deletes ndg963qz. The heuristic
      models this by assigning each available cz to at most one goal (greedy
      assignment).
    - pt4b08k4 objects that do not currently have any (qkdx0xe2 ?n54xx ?ts)
      facts are ignored since they cannot be moved with nm6lzp2x nor used in
      jql8txb3 (both require an existing qkdx0xe2 for the pt4b08k4).
    - If essential preconditions are missing from the state (e.g., no
      (of6wn8m4 ?ris) or ris has no qkdx0xe2 fact), the heuristic assigns a
      large finite penalty for that goal (so solvable states remain finite).

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic:
    - Stores the set of goal facts from the task.
    - Extracts the set of target ris objects from the goals (arguments of ne6sl557).
    - Builds the directed adjacency list for the lpmls56r relation from the
      task.static facts: this models the allowed forward moves for nm6lzp2x.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal ris (i.e., for each (ne6sl557 ?ris) that is not yet true in
    the current state) the heuristic estimates the minimal number of operator
    applications required as follows:

    1. Quick checks:
       - If (ne6sl557 ?ris) already holds in the state, cost = 0.
       - If (of6wn8m4 ?ris) is missing -> mark goal as unreachable and add a
         large finite penalty.
       - Extract the mz node ts_ris for ris from the state's (qkdx0xe2 ?ris ?ts)
         facts; if none -> unreachable penalty.

    2. Candidate cz selection:
       - Consider only cz (du770mtw) objects that currently have (ndg963qz ?cz)
         and also have (qkdx0xe2 ?cz ts_ris). f7y6nzq3 requires cz with that ts.
       - If there is no such cz -> unreachable penalty.

    3. For a given ris and a chosen cz (with ndg and qkdx0xe2 cz ts_ris),
       we must produce or reuse an n54xx (pt4b08k4) that satisfies
       (qkdx0xe2 ?n54xx ts_ris) and (fqyrzl30 ?n54xx ?cz).
       - If a fqyrzl30 linking an n54xx to cz already exists, we still must
         ensure the n54xx is at ts_ris (possibly moving it with nm6lzp2x).
       - If no such fqyrzl30 exists, we must move some n54xx to ts_ris
         (using nm6lzp2x repeated along lpmls56r) and then apply jql8txb3 to
         create the fqyrzl30. That costs: (distance in lpmls graph) +
         1 (jql8txb3) + 1 (f7y6nzq3).
       - If fqyrzl30 already exists and n54xx already at ts_ris, cost = 1
         (just f7y6nzq3). If fqyrzl30 exists but n54xx needs movement, cost =
         distance + 1.

    4. Compute minimal cost among all (cz, n54xx) combinations for this ris.

    5. Because f7y6nzq3 consumes the ndg flag of cz, a single cz can serve at
       most one goal. To model this, we perform a simple greedy assignment:
       repeatedly pick the (ris, cz) pair with smallest estimated cost,
       assign cz to that ris, and remove cz from available pool (repeat until
       all goals are assigned). If a ris has no feasible (cz, n54xx)
       combination, add the unreachable penalty for that ris.

    6. The returned heuristic value is the sum of the assigned costs over all
       unsatisfied goals. This is intentionally not strictly admissible but
       aims to provide a useful, cheaply computable estimate that takes into
       account movement along lpmls56r, the need to create fqyrzl30 where
       necessary, and the one-time consumption of ndg resources.

    Note on Finite Values
    ---------------------
    - For states that are solvable under the domain dynamics the heuristic
      returns a finite integer. For states that are provably impossible under
      the current facts (missing essential preconditions), the heuristic
      returns a large finite penalty (default 1000 per unreachable goal).
    """

    def __init__(self, task):
        """
        Extract static lpmls56r adjacency and goals from the task.

        - self.goals: set of goal fact strings (task.goals)
        - self.target_ris: set of ris object names that appear in ne6sl557 goals
        - self.lpmls_adj: dict mapping mz node -> list of successor mz nodes
                          (directed edges from static facts)
        """
        self.goals = set(task.goals)
        # Extract target ris objects from ne6sl557 goals
        self.target_ris = set()
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "ne6sl557" and len(parts) >= 2:
                self.target_ris.add(parts[1])

        # Build directed adjacency list for lpmls56r (static)
        self.lpmls_adj = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "lpmls56r" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.lpmls_adj.setdefault(a, []).append(b)

        # large finite penalty for unreachable goals
        self.unreachable_penalty = 1000

    def _bfs_distance(self, start, target):
        """
        Return shortest directed distance from start mz node to target mz node
        following lpmls56r edges. If start == target return 0. If unreachable,
        return None.
        """
        if start == target:
            return 0
        if start not in self.lpmls_adj:
            return None
        visited = {start}
        dq = deque([(start, 0)])
        while dq:
            node, d = dq.popleft()
            for succ in self.lpmls_adj.get(node, []):
                if succ == target:
                    return d + 1
                if succ not in visited:
                    visited.add(succ)
                    dq.append((succ, d + 1))
        return None

    def __call__(self, node):
        """
        Compute heuristic for state node.state (frozenset of fact strings).

        Returns integer estimated number of actions to satisfy all remaining
        ne6sl557 goals.
        """
        state = set(node.state)

        # If goal already reached, heuristic is zero.
        # However, we must check all goals in self.goals. If all already in state => 0.
        if self.goals <= state:
            return 0

        # Parse relevant facts from state
        qkdx_map = {}          # object -> set of mz nodes where (qkdx0xe2 obj ts)
        fq_map = {}            # (n54xx, cz) -> True if (fqyrzl30 n54xx cz) present
        ndg_set = set()        # set of cz with ndg963qz
        of6_set = set()        # set of ris with of6wn8m4
        ne6_set = set()        # set of ris with ne6sl557 already true

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "qkdx0xe2" and len(parts) >= 3:
                obj, ts = parts[1], parts[2]
                qkdx_map.setdefault(obj, set()).add(ts)
            elif pred == "fqyrzl30" and len(parts) >= 3:
                n54xx, cz = parts[1], parts[2]
                fq_map[(n54xx, cz)] = True
            elif pred == "ndg963qz" and len(parts) >= 2:
                ndg_set.add(parts[1])
            elif pred == "of6wn8m4" and len(parts) >= 2:
                of6_set.add(parts[1])
            elif pred == "ne6sl557" and len(parts) >= 2:
                ne6_set.add(parts[1])

        # Identify unsatisfied target ris objects that we must consider
        unsat_ris = [ris for ris in self.target_ris if ris not in ne6_set]

        # If no unsatisfied goals, heuristic = 0
        if not unsat_ris:
            return 0

        # Build list of candidate pt4b08k4 objects (n54xx) we can use:
        # those that have at least one qkdx0xe2 entry (so they can be moved/used).
        candidate_n54 = [obj for obj in qkdx_map.keys()]

        # Precompute for each ris its ts_ris (qkdx0xe2 ris ts). We treat ris with
        # no ts as unreachable.
        ris_ts = {}
        for ris in unsat_ris:
            ts_set = qkdx_map.get(ris, None)
            if not ts_set:
                ris_ts[ris] = None
            else:
                # If multiple ts exist, consider each; but typical cases have one.
                # We'll use the set and evaluate all possibilities by treating
                # multiple ts by exploring options (but in practice one).
                ris_ts[ris] = set(ts_set)

        # Build candidate list of cz that are currently available (ndg set)
        available_cz = set(ndg_set)

        # For each ris and each cz we will compute minimal cost to achieve ris
        # using that cz (assuming cz remains available). Store tuples (ris, cz, cost)
        options = []  # list of (ris, cz, cost)

        for ris in unsat_ris:
            ts_options = ris_ts.get(ris)
            # Quick invalid cases
            if ris not in of6_set or ts_options is None:
                # No of6wn8m4 or no qkdx0xe2 for ris -> unreachable
                continue

            # For each possible ts where ris sits, evaluate cz candidates
            for ts_ris in ts_options:
                for cz in list(available_cz):
                    # cz must have qkdx0xe2 cz ts_ris to be usable
                    if ts_ris not in qkdx_map.get(cz, set()):
                        continue

                    # For this ris and cz we need some n54xx such that after
                    # possibly moving it (nm6lzp2x) it is at ts_ris and either
                    # already fqyrzl30 n54xx cz exists or we will create it
                    # with one jql8txb3.

                    best_cost_for_pair = None
                    for n54 in candidate_n54:
                        # n54 must have qkdx0xe2 facts to be movable/usable
                        n54_ts_set = qkdx_map.get(n54, None)
                        if not n54_ts_set:
                            continue
                        # For every current ts_n for n54, compute distance to ts_ris
                        best_distance = None
                        for ts_n in n54_ts_set:
                            dist = self._bfs_distance(ts_n, ts_ris)
                            if dist is None:
                                # not reachable from this ts_n
                                continue
                            if best_distance is None or dist < best_distance:
                                best_distance = dist
                        if best_distance is None:
                            # this n54 cannot reach ts_ris
                            continue

                        # Determine if fqyrzl30 n54 cz already exists
                        jql_needed = 0 if (n54, cz) in fq_map else 1

                        # Cost = moves (nm6lzp2x count) + (jql if needed) + f7y6nzq3 (1)
                        cost = best_distance + jql_needed + 1

                        if best_cost_for_pair is None or cost < best_cost_for_pair:
                            best_cost_for_pair = cost

                    if best_cost_for_pair is not None:
                        options.append((ris, cz, best_cost_for_pair))

        # If there are no options computed for any ris, then all unsatisfied goals
        # are unreachable under current facts -> return sum of unreachable penalties
        if not options:
            return len(unsat_ris) * self.unreachable_penalty

        # Greedy assignment of cz resources to ris goals:
        # Repeatedly select the smallest-cost (ris, cz) pair, assign cz to ris,
        # and remove all other options that use that cz or that ris.
        total_cost = 0
        remaining_options = list(options)
        remaining_ris = set(unsat_ris)
        remaining_cz = set(available_cz)

        # If a ris had no options at all (not in any option tuple) treat as unreachable
        ris_with_options = {opt[0] for opt in remaining_options}
        for ris in remaining_ris - ris_with_options:
            total_cost += self.unreachable_penalty
        remaining_ris &= ris_with_options

        # Greedy loop
        while remaining_ris:
            # Filter remaining options to ones with ris in remaining_ris and cz in remaining_cz
            valid_opts = [opt for opt in remaining_options if opt[0] in remaining_ris and opt[1] in remaining_cz]
            if not valid_opts:
                # Some remaining ris are unreachable with remaining cz
                total_cost += len(remaining_ris) * self.unreachable_penalty
                break
            # pick minimal cost option
            ris_choice, cz_choice, cost_choice = min(valid_opts, key=lambda x: x[2])
            total_cost += cost_choice
            # remove assigned ris and cz
            remaining_ris.discard(ris_choice)
            remaining_cz.discard(cz_choice)
            # Remove all options that involve this ris or cz
            remaining_options = [opt for opt in remaining_options if opt[0] != ris_choice and opt[1] != cz_choice]

        return total_cost