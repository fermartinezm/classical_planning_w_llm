from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(qkdx0xe2 n16nw8x1 w5eewe3t)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
        This heuristic estimates the minimum number of actions required to achieve the goal
        (which requires all objects of type `wqid8mog` to have predicate `ne6sl557` true).
        The main challenge involves:
         - Moving certain `pt4b08k4` objects along a chain of `mz6r1d30` objects (via `nm6lzp2x`),
         - Associating `pt4b08k4` objects with `du770mtw` objects (via `jql8txb3`),
         - Finally performing the action `f7y6nzq3` on pairs `(wqid8mog, du770mtw, pt4b08k4, mz6r1d30)` to achieve "ne6sl557".

    Assumptions:
        - The goal requires all relevant `wqid8mog` objects to have `ne6sl557` true.
        - `f7y6nzq3` achieves `ne6sl557`, requires:
            * the corresponding `du770mtw` object to be `ndg963qz` (active),
            * the `pt4b08k4` object to be associated with that `du770mtw` via `fqyrzl30`,
            * both the `wqid8mog` and the `pt4b08k4` to be at the same `mz6r1d30` location (`qkdx0xe2`),
            * the `wqid8mog` to have predicate `of6wn8m4` true.
        - To achieve the goal, we must move `pt4b08k4` objects along stationary chains of `mz6r1d30` objects (`lpmls56r`) to the correct location, 
          associate them with the right `du770mtw`, and then use `f7y6nzq3`.

    Heuristic Initialization:
        - Extract the chain-structure of `mz6r1d30` objects from static facts `(lpmls56r ?from ?to)`.
        - Extract all `wqid8mog`, `du770mtw`, and `pt4b08k4` objects from facts.
        - Identify goal `wqid8mog` objects that must have `(ne6sl557 ?wqid8mog)`.
        - Represent the `lpmls56r` relation as a graph to estimate distances between `mz6r1d30` locations.

    Step-By-Step Thinking for Computing Heuristic:
        1. Identify how many `wqid8mog` objects in the goal are not yet `ne6sl557` in the current state. Each requires at least 1 `f7y6nzq3` action.
        2. For each such goal `wqid8mog`:
            - Find its current location `mz6r1d30` if it has `(qkdx0xe2 wqid8mog mz6r1d30)`.
            - Find a candidate `du770mtw` object holding `(ndg963qz du770mtw)` (active) that can enable `f7y6nzq3`.
            - For that `du770mtw`, find the associated `pt4b08k4` object (via `(fqyrzl30 pt4b08k4 du770mtw)`).
            - Determine the location of that `pt4b08k4` (which is the location of the `(qkdx0xe2 pt4b08k4 mz6r1d30)` fact).
            - If `pt4b08k4` is not at the same location as the `wqid8mog`, estimate how many `nm6lzp2x` actions are required to move `pt4b08k4` to the `wqid8mog`'s location — this equals the distance between locations in the chain.
            - If the `pt4b08k4` is not yet associated with the `du770mtw`, include 1 `jql8txb3` action to associate.
            - Include 1 `f7y6nzq3` action for finalizing `ne6sl557`.
        3. Sum all these estimated costs for each goal `wqid8mog` not yet achieved.
        4. Return the total as the heuristic value. It is 0 if all `(ne6sl557 ?wqid8mog)` are satisfied.

    This heuristic is not admissible but aims to be informed and efficiently computable by:
     - Using distances on the static `lpmls56r` graph for estimating movement actions.
     - Counting minimal necessary steps per `wqid8mog` goal fact.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Build forward graph for lpmls56r (links between mz6r1d30 objects)
        self.graph = {}
        for fact in self.static:
            if match(fact, "lpmls56r", "*", "*"):
                parts = get_parts(fact)
                frm, to = parts[1], parts[2]
                self.graph.setdefault(frm, set()).add(to)
                # Also build reversed edges to allow bidirectional search
                self.graph.setdefault(to, set()).add(frm)

        # Extract all objects by type from facts (approximate from facts in static + goals)
        self.mz6r1d30 = set()
        self.d11dxhbr = set()
        self.pt4b08k4 = set()
        self.wqid8mog = set()
        self.du770mtw = set()
        self.pt4b08k4_by_fqyrzl = {}    # pt4b08k4 -> set of du770mtw associated via fqyrzl30
        self.ndg963qz_du770mtw = set()  # active du770mtw (ndg963qz ?du770mtw)
        self.goal_ne6sl557_wqid8mog = set()

        # Helper collections for quick lookup:
        # These track current facts that won't change in heuristic call (except goals)
        # We'll fill these in __call__ since state changes

        # Precompute goal wqid8mog for ne6sl557
        for fact in self.goals:
            if match(fact, "ne6sl557", "*"):
                self.goal_ne6sl557_wqid8mog.add(get_parts(fact)[1])

    def shortest_distance(self, start, end):
        """
        Compute shortest undirected distance between two mz6r1d30 nodes using BFS.
        Return integer length or None if unreachable.
        """
        if start == end:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            neighbors = self.graph.get(current, set())
            for nbr in neighbors:
                if nbr == end:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        state = node.state

        # 1) Check how many goal wqid8mog have ne6sl557 in state
        achieved = {get_parts(f)[1] for f in state if match(f, "ne6sl557", "*")}
        needed = self.goal_ne6sl557_wqid8mog - achieved

        if not needed:
            # All goals met
            return 0

        # 2) Extract current facts for heuristic computation:

        # qkdx0xe2 facts: map object -> location (mz6r1d30)
        qkdx0xe2_map = {}
        for fact in state:
            if match(fact, "qkdx0xe2", "*", "*"):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                qkdx0xe2_map[obj] = loc
                # Infer types by clustering known vars (rough guess)
                # We'll use object name prefixes or presence in facts to infer later if needed

        # fqyrzl30 facts: pt4b08k4 associated to du770mtw
        fqyrzl30_map = {}
        for fact in state:
            if match(fact, "fqyrzl30", "*", "*"):
                parts = get_parts(fact)
                pt4b08k4_obj, du770mtw_obj = parts[1], parts[2]
                fqyrzl30_map.setdefault(pt4b08k4_obj, set()).add(du770mtw_obj)

        # ndg963qz facts: active du770mtw
        ndg963qz_set = set()
        for fact in state:
            if match(fact, "ndg963qz", "*"):
                parts = get_parts(fact)
                ndg963qz_set.add(parts[1])

        # of6wn8m4 facts: wqid8mog with this predicate
        of6wn8m4_set = set()
        for fact in state:
            if match(fact, "of6wn8m4", "*"):
                parts = get_parts(fact)
                of6wn8m4_set.add(parts[1])

        # lpmls56r in static (chain structure already stored)

        total_estimated_cost = 0

        # For each needed wqid8mog goal:
        for wobj in needed:
            # Preconditions for f7y6nzq3:
            # - ndg963qz ?cz2xr  where ?cz2xr is du770mtw
            # - qkdx0xe2 ?n54xx ?ts6t4 (the pt4b08k4 at the location)
            # - qkdx0xe2 ?ris62 ?ts6t4 (the wqid8mog at same location)
            # - fqyrzl30 ?n54xx ?cz2xr (pt4b08k4 associated to du770mtw)
            # - of6wn8m4 ?ris62 (wqid8mog has predicate)
            # - ndg963qz ?cz2xr (du770mtw active)
            # - qkdx0xe2 wqid8mog loc and pt4b08k4 loc equal

            # Current location of wobj
            wobj_loc = qkdx0xe2_map.get(wobj)
            if wobj_loc is None:
                # If wqid8mog is not at known location, we pessimistically assume some cost (cannot apply f7y6nzq3)
                # We add a large cost (e.g. 5) to encourage actions that fix this
                total_estimated_cost += 5
                continue

            # of6wn8m4 must hold on wobj to apply f7y6nzq3
            if wobj not in of6wn8m4_set:
                # Must apply action or actions to get wobj into of6wn8m4 status,
                # but no action achieves of6wn8m4 directly (only f7y6nzq3 deletes of6wn8m4)
                # Therefore, likely impossible or cost = 0 (ignore since task yields it)
                # To be safe, add minimal penalty (1) since f7y6nzq3 requires it
                total_estimated_cost += 1

            # Search for a du770mtw object that is active (ndg963qz) and suitable
            # and a pt4b08k4 associated with it.

            # Among ndg963qz active du770mtw, find one with some fqyrzl30 pt4b08k4 associated
            # so that pt4b08k4 can be moved to wobj_loc to apply f7y6nzq3.

            best_cost_for_goal = None

            for du_obj in ndg963qz_set:
                # For each pt4b08k4 associated with du_obj:
                associated_pts = [pt for pt, dus in fqyrzl30_map.items() if du_obj in dus]

                if not associated_pts:
                    # Might need one jql8txb3 to associate, so consider all pt4b08k4 to associate
                    # Heuristic: assume cost 1 to associate + movement cost
                    # Try all pt4b08k4 in domain (approximate from qkdx0xe2 for pt4b08k4 objects)
                    pt_candidates = [
                        obj for obj in qkdx0xe2_map if obj not in ndg963qz_set and obj != wobj
                    ]
                    if not pt_candidates:
                        continue
                    associated_pts = pt_candidates  # consider all pt4b08k4 here

                    need_association = True
                else:
                    need_association = False

                for pt_obj in associated_pts:
                    pt_loc = qkdx0xe2_map.get(pt_obj)
                    if pt_loc is None:
                        # unknown location, add high cost penalty
                        move_cost = 5
                    else:
                        dist = self.shortest_distance(pt_loc, wobj_loc)
                        move_cost = dist if dist is not None else 5

                    # Cost summary:
                    # - 1 for association if needed (jql8txb3)
                    # - move_cost for nm6lzp2x actions to move pt4b08k4 to wobj location
                    # - 1 for final f7y6nzq3 to achieve ne6sl557
                    cost = 1 + move_cost + 1
                    if need_association:
                        cost += 1  # the extra association step jql8txb3

                    if best_cost_for_goal is None or cost < best_cost_for_goal:
                        best_cost_for_goal = cost

            if best_cost_for_goal is None:
                # No du770mtw active or no pt4b08k4 to associate, add penalty (e.g. 7)
                total_estimated_cost += 7
            else:
                total_estimated_cost += best_cost_for_goal

        return total_estimated_cost
