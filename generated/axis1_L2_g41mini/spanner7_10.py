from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(qkdx0xe2 n16nw8x1 mpwrued2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    The heuristic estimates the number of actions required to achieve the goal
    condition `(ne6sl557 ?ris62)` for one or several ?ris62 objects of type wqid8mog,
    i.e., to "activate" all goal items of that form. Activation corresponds roughly to
    performing the `f7y6nzq3` action, which requires specific preconditions to hold.

    Assumptions
    -----------
    - The goal always requires all given objects (type wqid8mog) to hold the predicate `ne6sl557`.
    - An object ?ris62 of type wqid8mog can only become ne6sl557 after applying `f7y6nzq3`.
    - To apply `f7y6nzq3` on ?ris62, a ?cz2xr (du770mtw) must satisfy predicate ndg963qz,
      the ?ris62 must currently hold of6wn8m4, and a linking ?ts6t4 (mz6r1d30) must be
      connected correctly between ?ris62 and some ?n54xx (pt4b08k4) which itself must hold
      fqyrzl30 with ?cz2xr.
    - The `f7y6nzq3` action deletes the of6wn8m4 for that ?ris62 and ndg963qz for that ?cz2xr.
    - The action `jql8txb3` moves an fqyrzl30 relation from ?cz2xr to ?n54xx, but only if both share qkdx0xe2 with the same ?ts6t4.
    - The action `nm6lzp2x` moves the qkdx0xe2 relation for a pt4b08k4 object along the lpmls56r chain, which
      appears to represent a path or ordering on mz6r1d30 objects.
    - The static relation lpmls56r forms chains between mz6r1d30 objects, used to "walk" qkdx0xe2 facts.

    Heuristic Initialization
    ------------------------
    - Extract the static chains formed by lpmls56r, interpreting them as a directed graph among mz6r1d30 nodes.
    - Extract the goal objects of type wqid8mog that must become ne6sl557.
    - Store all static facts for efficient lookup.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal object ?ris62 of type wqid8mog:
    1. Check if (ne6sl557 ?ris62) already holds in the state. If yes, no cost for this object.
    2. Otherwise, the object needs to be activated via `f7y6nzq3`.
       Preconditions:
       - A (ndg963qz ?cz2xr) fact: a du770mtw object with ndg963qz true (think of it as a "free" or "available" unit).
       - The object ?ris62 must currently hold (of6wn8m4 ?ris62).
       - A linking ?ts6t4 must be qkdx0xe2 connected both to ?ris62 and to some ?n54xx (pt4b08k4).
       - That ?n54xx must hold (fqyrzl30 ?n54xx ?cz2xr).
    3. To fulfill this, we must potentially:
       - Transport fqyrzl30 facts among du770mtw objects (?cz2xr).
       - Move qkdx0xe2 connections along lpmls56r chains from ?n54xx to ?ris62 or vice versa via nm6lzp2x.
       - Find a ?cz2xr with ndg963qz that can participate.
    4. Approximate cost by:
       - Counting how far the qkdx0xe2 connection on pt4b08k4 can move along lpmls56r chains to connect with the ?ts6t4 linked to both.
       - Count whether ndg963qz is available or needs to be re-established (treated as 1 action if absent).
       - Count one action for f7y6nzq3 itself.
    5. Sum costs for all goal objects; since their activations are independent, sum is a good estimate.
    6. Since the heuristic need not be admissible, this approximation is focused on a lower bound on required connection moves and activations.
    """

    def __init__(self, task):
        """Initialize the heuristic and extract static chains and goal objects."""
        self.goals = task.goals
        self.static = task.static

        # Extract the chain graph from static lpmls56r relations: directed edges (from -> to)
        self.chain_graph = {}
        for fact in self.static:
            if match(fact, "lpmls56r", "*", "*"):
                _, from_node, to_node = get_parts(fact)
                if from_node not in self.chain_graph:
                    self.chain_graph[from_node] = set()
                self.chain_graph[from_node].add(to_node)

        # Extract all goal ris62 objects that must be ne6sl557
        self.goal_ris62 = []
        for fact in self.goals:
            if match(fact, "ne6sl557", "*"):
                self.goal_ris62.append(get_parts(fact)[1])

    def _find_shortest_chain_distance(self, start, goal):
        """
        Compute shortest distance from start mz6r1d30 node to goal mz6r1d30 node along chain_graph.
        Return integer distance if reachable, else None.
        """
        if start == goal:
            return 0
        visited = set()
        frontier = [(start, 0)]
        while frontier:
            current, dist = frontier.pop(0)
            if current in visited:
                continue
            visited.add(current)
            neighbors = self.chain_graph.get(current, set())
            for nbr in neighbors:
                if nbr == goal:
                    return dist + 1
                frontier.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        """Compute estimated number of actions needed from the current state to reach the goal."""
        state = node.state

        # If goal reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Helper: extract sets of facts by predicate for faster queries
        def facts_for_predicate(predicate):
            return {fact for fact in state if fact.startswith("(" + predicate + " ")}

        qkdx_facts = facts_for_predicate("qkdx0xe2")
        fqyrzl30_facts = facts_for_predicate("fqyrzl30")
        ndg963qz_facts = facts_for_predicate("ndg963qz")
        of6wn8m4_facts = facts_for_predicate("of6wn8m4")

        # Extract all du770mtw objects with ndg963qz (available for f7y6nzq3)
        available_cz2xr = set(get_parts(fact)[1] for fact in ndg963qz_facts)

        # Extract mapping from pt4b08k4 object to the set of mz6r1d30 it is connected to via qkdx0xe2
        pt4b08k4_to_mz6r1d30 = {}
        # Also build reverse map for quick lookup from mz6r1d30 to pt4b08k4
        mz6r1d30_to_pt4b08k4 = {}
        for fact in qkdx_facts:
            _, obj1, obj2 = get_parts(fact)
            # Determine types by heuristic guess from domain typing:
            # qkdx0xe2 has two arguments: ?n54xx - d11dxhbr, ?ts6t4 - mz6r1d30
            # pt4b08k4 is subtype of d11dxhbr, but also d11dxhbr covers du770mtw.
            # Use name patterns: pt4b08k4 objects appear in position 1 when first argument is pt4b08k4
            if obj1.startswith("xd0u8kgj") or obj1[0].isalpha():  # heuristic fallback
                # we need pt4b08k4 -> mz6r1d30 mapping
                pt4b08k4_to_mz6r1d30.setdefault(obj1, set()).add(obj2)
                mz6r1d30_to_pt4b08k4.setdefault(obj2, set()).add(obj1)

        # Extract fqyrzl30 facts mapping pt4b08k4 to du770mtw
        pt4b08k4_to_du770mtw = {}
        for fact in fqyrzl30_facts:
            _, pt4b08k4_obj, du770mtw_obj = get_parts(fact)
            pt4b08k4_to_du770mtw.setdefault(pt4b08k4_obj, set()).add(du770mtw_obj)

        # Extract of6wn8m4 wqid8mog objects that still need activation (goal objects)
        needed_obj = [obj for obj in self.goal_ris62 if f"(ne6sl557 {obj})" not in state]

        total_cost = 0

        for ris62 in needed_obj:
            # Must activate ris62: cost at least 1 for f7y6nzq3 action itself
            # Preconditions depend on finding:
            # - cz2xr in available_cz2xr
            # - ts6t4 (mz6r1d30) with (qkdx0xe2 ris62 ts6t4) and (qkdx0xe2 n54xx ts6t4)
            #   and (fqyrzl30 n54xx cz2xr)
            # - (of6wn8m4 ris62) must hold

            # Check presence of of6wn8m4 for ris62
            of6wn8m4_here = f"(of6wn8m4 {ris62})" in of6wn8m4_facts
            if not of6wn8m4_here:
                # Must move of6wn8m4 on ris62 to enable f7y6nzq3
                # Count 1 cost for making of6wn8m4 happen (underapproximation)
                total_cost += 1

            # Get all ts6t4 connected to ris62 via qkdx0xe2
            ts6t4_candidates = set()
            for fact in qkdx_facts:
                p, arg1, arg2 = get_parts(fact)
                if arg1 == ris62:
                    ts6t4_candidates.add(arg2)

            if not ts6t4_candidates:
                # No direct qkdx0xe2 link from ris62 to any ts6t4, 
                # must move qkdx0xe2 via nm6lzp2x actions, count minimal cost as 1
                total_cost += 1
                # Treat ts6t4_candidates as empty, so heuristic remains finite
                ts6t4_candidates = set()

            # Determine minimal distance to connect ts6t4 and n54xx s.t. 
            # qkdx0xe2(n54xx, ts6t4) and fqyrzl30(n54xx, cz2xr)
            # We'll approximate by finding minimal chain distance between ts6t4 and
            # any mz6r1d30 linked to fqyrzl30(n54xx, cz2xr) where cz2xr ∈ available_cz2xr

            min_chain_cost = None

            # Build inverse map from fqyrzl30: all n54xx with loading to some cz2xr with ndg963qz
            n54xx_candidates = set()
            for fact in fqyrzl30_facts:
                _, n54xx_obj, cz2xr_obj = get_parts(fact)
                if cz2xr_obj in available_cz2xr:
                    n54xx_candidates.add(n54xx_obj)

            # For each candidate ts6t4 and n54xx, find qkdx0xe2 connection to ts6t4 for n54xx along chain
            # We check if there exists a link from n54xx to ts6t4 or ts6t4 to n54xx with chain moves
            for ts6t4 in ts6t4_candidates:
                for n54xx in n54xx_candidates:
                    # Check if qkdx0xe2(n54xx, ts6t4) directly exists
                    direct = f"(qkdx0xe2 {n54xx} {ts6t4})" in qkdx_facts
                    if direct:
                        chain_dist = 0
                    else:
                        # Check shortest chain distance from n54xx to ts6t4
                        dist1 = self._find_shortest_chain_distance(n54xx, ts6t4)
                        # or from ts6t4 to n54xx
                        dist2 = self._find_shortest_chain_distance(ts6t4, n54xx)

                        # Choose minimal available distance
                        candidates = [d for d in [dist1, dist2] if d is not None]
                        if not candidates:
                            continue
                        chain_dist = min(candidates)

                    if min_chain_cost is None or chain_dist < min_chain_cost:
                        min_chain_cost = chain_dist

            if min_chain_cost is None:
                # No connection found, estimate heuristic cost as 2 (at least 1 move + 1 action)
                min_chain_cost = 2

            # Cost:
            # - chain moves via nm6lzp2x (min_chain_cost actions)
            # - one f7y6nzq3 action
            # - if no ndg963qz available cz2xr, count 1 extra action (we check that below)

            # Check if any ndg963qz facts exist
            if not available_cz2xr:
                # No free cz2xr with ndg963qz - must pay at least 1
                total_cost += 1

            total_cost += min_chain_cost + 1  # +1 for f7y6nzq3 action

        return total_cost
