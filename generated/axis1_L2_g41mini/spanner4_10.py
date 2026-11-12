from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(qkdx0xe2 n16nw8x1 mpwrued2)".
    - `args`: The pattern, wildcards `*` allowed.
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (du770mtw).

    Summary:
    This heuristic estimates the number of actions needed to reach a goal state
    where all goal facts (typically `(ne6sl557 ?ris62)`) are satisfied.
    The key challenge is to produce an efficiently computable estimate that
    guides the greedy best-first search effectively, even if it is not admissible.

    Assumptions:
    - The main goal predicate is `(ne6sl557 ?ris62)` for certain objects of type `wqid8mog`.
    - Achieving `(ne6sl557 ?ris62)` requires utilization of the `(of6wn8m4 ?ris62)` fact and `(ndg963qz ?cz2xr)`
      (acts like some kind of resource/flag).
    - Objects of type `mz6r1d30` are linked as a chain via `(lpmls56r ?from ?to)`.
    - `(qkdx0xe2 ?entity ?mz6r1d30)` indicates some kind of location or assignment of an object to a location.
    - Actions:
      - `nm6lzp2x` moves a `pt4b08k4` object along the `mz6r1d30` chain (changing the `qkdx0xe2` relation).
      - `jql8txb3` acts on a `du770mtw` and a `pt4b08k4`, redistributing facts `(fqyrzl30)` and altering `(qkdx0xe2)`.
      - `f7y6nzq3` is the only action that achieves `(ne6sl557 ?ris62)` goal predicates, requiring:
        - `(ndg963qz ?cz2xr)` present,
        - a `pt4b08k4` linked to that `du770mtw` by `(fqyrzl30)`,
        - `(of6wn8m4 ?ris62)` true,
        - and `(qkdx0xe2 ?ris62 ?ts6t4)`.

    Heuristic Initialization:
    - Extract goals (all `(ne6sl557 ?ris62)`) objects of type `wqid8mog` to track.
    - Extract the `lpmls56r` chain to quickly compute distances between locations of type `mz6r1d30`.
      We build a graph (directed edges) to represent this chain.
    - Extract static `(lpmls56r ?from ?to)` facts to build the chain graph.
    
    Step-By-Step Thinking for Computing Heuristic:
    1. For every goal object `?ris62` (wqid8mog) to achieve `(ne6sl557 ?ris62)`, if already true in state,
       heuristic contribution from that object is zero.
    2. Otherwise, estimate the minimal cost to enable `(ne6sl557 ?ris62)`:
       This requires:
       - `(of6wn8m4 ?ris62)` present in state (otherwise cannot achieve goal, heuristic large).
       - Find a `du770mtw` object `?cz2xr` for which `(ndg963qz ?cz2xr)` is true (needed for action `f7y6nzq3`).
       - Find a `pt4b08k4` object `?n54xx` currently linked to `?cz2xr` via `(fqyrzl30 ?n54xx ?cz2xr)`.
       - `?n54xx` must be at some `mz6r1d30` location `(qkdx0xe2 ?n54xx ?ts6t4)` that also satisfies `(qkdx0xe2 ?ris62 ?ts6t4)`.
         If not, estimate cost to move `?n54xx` to that location by traversing `(lpmls56r ...)` chain.
       - `?ris62` itself must be at some `mz6r1d30` location indicated by `(qkdx0xe2 ?ris62 ?ts6t4)`.
    3. The minimal cost for each goal object:
       - 1 action to execute `f7y6nzq3` (the goal-achieving action).
       - Plus estimated minimal number of moves to get `?n54xx` to a required `mz6r1d30` location matching `?ris62`'s location,
         using `nm6lzp2x` actions to move along the `lpmls56r` chain.
       - If `?n54xx` is not already linked via `(fqyrzl30 ?n54xx ?cz2xr)`, 1 action for `jql8txb3` to connect it (heuristic).
       - If multiple choices exist, choose the minimal cost path (minimum over all suitable `du770mtw` and `pt4b08k4`).
    4. Summing over all goal objects, return total heuristic cost estimate.
    5. If some conditions cannot be matched, return a large heuristic to discourage.

    This heuristic encourages resolving one goal object at a time,
    taking into account distances in the `mz6r1d30` chain and
    the necessary actions to prepare the required preconditions.
    """

    def __init__(self, task):
        self.goals = task.goals

        # Extract goal wqid8mog objects needing to satisfy (ne6sl557 ?ris62).
        self.goal_ris62_objects = set()
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "ne6sl557" and len(parts) == 2:
                self.goal_ris62_objects.add(parts[1])

        # Build a directed graph from static lpmls56r chain facts: mz6r1d30 -> mz6r1d30
        self.chain_graph = {}
        for fact in task.static:
            if match(fact, "lpmls56r", "*", "*"):
                _, from_loc, to_loc = get_parts(fact)
                self.chain_graph.setdefault(from_loc, set()).add(to_loc)
        # Precompute shortest distances in the chain graph for efficiency.
        # We use BFS to get distances between any two chain nodes.
        # Cache distances in a dict: distances[src][tgt] = min steps or None.
        self.distances = {}
        for node in self.chain_graph.keys():
            self.distances[node] = self._bfs_distances(node)

        # Gather static du770mtw objects (keys for ndg963qz and fqyrzl30)
        self.du_objects = {fact_parts[1] for fact_parts in (get_parts(f) for f in task.static | task.initial_state)
                           if fact_parts[0] == "ndg963qz"}

    def _bfs_distances(self, start):
        from collections import deque
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.chain_graph.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        state = node.state

        # Quick check if goal achieved (heuristic = 0)
        if self.goals <= state:
            return 0

        # Extract helpful sets from state:
        # Sets for:
        # - objects with (of6wn8m4 ?ris62)
        of6wn8m4_objects = {get_parts(fact)[1] for fact in state if match(fact, "of6wn8m4", "*")}
        # - objects with (ndg963qz ?cz2xr)
        ndg963qz_objects = {get_parts(fact)[1] for fact in state if match(fact, "ndg963qz", "*")}
        # - (fqyrzl30 ?n54xx ?cz2xr) link
        fqyrzl30_links = {}
        for fact in state:
            if match(fact, "fqyrzl30", "*", "*"):
                _, n54xx, cz2xr = get_parts(fact)
                fqyrzl30_links.setdefault(cz2xr, set()).add(n54xx)
        # - (qkdx0xe2 ?entity ?mz6r1d30)
        # Store locations of all entities
        loc_of_entity = {}
        for fact in state:
            if match(fact, "qkdx0xe2", "*", "*"):
                _, entity, loc = get_parts(fact)
                loc_of_entity.setdefault(entity, set()).add(loc)

        total_estimated_cost = 0
        large_penalty = 1000  # Large heuristic cost for unreachable/goals not satisfiable.

        # Process each goal (ne6sl557 ?ris62)
        for ris62 in self.goal_ris62_objects:
            # If goal already achieved for this ris62, zero contribution.
            if f"(ne6sl557 {ris62})" in state:
                continue

            # Must have (of6wn8m4 ris62)
            if ris62 not in of6wn8m4_objects:
                # Cannot achieve goal for this ris62 now, penalize high
                total_estimated_cost += large_penalty
                continue

            # ris62 location(s)
            ris_locations = loc_of_entity.get(ris62)
            if not ris_locations:
                # Without location of ris62, heuristic unknown, penalize
                total_estimated_cost += large_penalty
                continue

            # Find candidate du770mtw objects with ndg963qz true
            if not ndg963qz_objects:
                total_estimated_cost += large_penalty
                continue

            minimal_cost_for_this_goal = large_penalty

            # For each candidate du770mtw with ndg963qz true
            for cz2xr in ndg963qz_objects:
                # pt4b08k4 objects currently linked to cz2xr via fqyrzl30
                linked_n54xxs = fqyrzl30_links.get(cz2xr, set())

                # If no pt4b08k4 linked, at least one jql8txb3 needed.
                jql8txb3_needed = len(linked_n54xxs) == 0

                # For each pt4b08k4 that could be linked (including no one linked)
                # We estimate cost:
                # - 1 action for f7y6nzq3 to set the goal pred
                # - optionally 1 action for jql8txb3 if linking needed
                # - number of nm6lzp2x moves to move pt4b08k4 to location matching ris62 location

                # If none linked, consider "virtual" single pt4b08k4 needing linking and moving:
                if jql8txb3_needed:
                    candidate_n54xxs = loc_of_entity.keys()  # any pt4b08k4 possible
                else:
                    candidate_n54xxs = linked_n54xxs

                # For each candidate pt4b08k4 object
                for n54xx in candidate_n54xxs:
                    # Locations of n54xx
                    n54xx_locs = loc_of_entity.get(n54xx)
                    if not n54xx_locs:
                        # Cannot estimate without location, skip
                        continue

                    # We want to move n54xx to a location that matches ris62 location.
                    # Since both are mz6r1d30 typed locations, we try to find distance between
                    # any n54xx location and any ris62 location on chain.

                    # Compute minimal distance over all location pairs
                    min_dist = None
                    for start in n54xx_locs:
                        for goal_loc in ris_locations:
                            dist_dict = self.distances.get(start)
                            if dist_dict:
                                dist = dist_dict.get(goal_loc)
                                if dist is not None:
                                    if min_dist is None or dist < min_dist:
                                        min_dist = dist

                    # If no distance found, try reverse (if chain is unidirectional)
                    if min_dist is None:
                        # Try reverse direction with BFS starting from ris62 locations
                        for goal_loc in ris_locations:
                            dist_dict_rev = self.distances.get(goal_loc)
                            if dist_dict_rev:
                                for start in n54xx_locs:
                                    dist = dist_dict_rev.get(start)
                                    if dist is not None:
                                        # if needed, we can consider reverse mv cost higher since chain is directed,
                                        # but we count same here.
                                        if min_dist is None or dist < min_dist:
                                            min_dist = dist

                    if min_dist is None:
                        # If no path in chain, penalize large for this candidate
                        continue

                    # Actions needed:
                    # - number of moves: nm6lzp2x actions = min_dist
                    # - if jql8txb3 needed (linking), +1
                    # - goal action f7y6nzq3 = 1

                    cost = min_dist + 1 + (1 if jql8txb3_needed else 0)

                    if cost < minimal_cost_for_this_goal:
                        minimal_cost_for_this_goal = cost

            # If minimal cost remains large_penalty, add just this penalty
            total_estimated_cost += minimal_cost_for_this_goal

        return total_estimated_cost
