from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate obj1 obj2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (du770mtw).

    Summary:
    This heuristic estimates the number of actions required to achieve the goal conditions
    of the secret_domain. The main goal predicate is (ne6sl557 ?ris62), which marks some
    objects of type wqid8mog as "completed" or "goal achieved". The heuristic focuses on
    how many of these goal objects have not yet been "activated" by the f7y6nzq3 action.

    Assumptions:
    - The goal is always a conjunction of ne6sl557 predicates on wqid8mog objects.
    - The only way to achieve (ne6sl557 ?ris62) is via the action f7y6nzq3.
    - f7y6nzq3 requires:
      -- A "du770mtw" object ?cz2xr with (ndg963qz ?cz2xr) true (a sort of resource or state)
      -- An object ?n54xx of type pt4b08k4 that satisfies (fqyrzl30 ?n54xx ?cz2xr)
      -- An object ?ts6t4 of type mz6r1d30 such that (qkdx0xe2 ?n54xx ?ts6t4) and (qkdx0xe2 ?ris62 ?ts6t4)
      -- ?ris62 must currently have (of6wn8m4 ?ris62)
    - The qkdx0xe2 predicate is movable via the nm6lzp2x action to "shift" an object's qkdx0xe2 relation along the static chain defined by lpmls56r.
    - Objects of type du770mtw lose the (ndg963qz) predicate after f7y6nzq3, which means they must be refreshed or restored (but no action for that observed here).

    Heuristic Initialization:
    - Extract the set of goal wqid8mog objects that must hold (ne6sl557 ?ris62).
    - Extract static chains of mz6t4 objects from lpmls56r facts to represent adjacency for qkdx0xe2 shifting.
    - Extract all du770mtw objects that can hold the ndg963qz predicate.
    - Extract facts about which pt4b08k4 objects are linked to du770mtw via fqyrzl30.
    - Extract initial placement facts of qkdx0xe2 predicate.
    - Extract which wqid8mog objects currently have the of6wn8m4 predicate (required by f7y6nzq3).

    Step-By-Step Thinking for Computing Heuristic:
    1. Count how many goal wqid8mog objects DO NOT have (ne6sl557 ?ris62) yet — these require processing.
       If zero, heuristic = 0.

    2. For each goal wqid8mog object not achieved:
       - Check if (of6wn8m4 ?ris62) is true; if not, it has to be made true by somehow restoring it,
         but no direct action for that found here, so we assume it costs at least 1 action to "refresh" it.

       - Find the current mz6t4 location(s) of the wqid8mog object (via qkdx0xe2 predicate).

       - For f7y6nzq3 to apply, this mz6t4 must coincide with one linked to some pt4b08k4 object ?n54xx which in turn is linked (fqyrzl30) to some du770mtw object ?cz2xr that holds ndg963qz.

       - Compute minimal number of nm6lzp2x actions to move qkdx0xe2 of ?n54xx to the mz6t4 where the wqid8mog object currently is;
         this involves walking along the static chain lpmls56r from the current mz6t4 to target mz6t4.

       - Check if the du770mtw object ?cz2xr with ndg963qz matching the pt4b08k4 ?n54xx exists.

       - One f7y6nzq3 action can then be applied to produce ne6sl557 and lose ndg963qz from ?cz2xr (making it unavailable until possibly restored).

    3. Summing up estimated costs:
       - Each missing goal wqid8mog requires:
         (a) 0 or 1 action if of6wn8m4 ?ris62 is missing (assumed 1 action cost),
         (b) minimal qkdx0xe2 shifts to align ?n54xx location to ?ri62 location (number of nm6lzp2x actions),
         (c) 1 f7y6nzq3 action to produce ne6sl557,
         (d) plus 1 extra action approximating possible needing to "refresh" ndg963qz (since it gets removed after f7y6nzq3).

    4. Return the total estimated action count as heuristic value.

    This heuristic is not admissible but attempts to estimate the minimum steps considering required shifts and applications of f7y6nzq3 actions.

    It uses static facts for adjacency (lpmls56r) to measure distances between mz6t4 objects,
    and matching of pt4b08k4 and du770mtw objects linked by fqyrzl30 and ndg963qz predicates.
    """

    def __init__(self, task):
        # Goals: set of wqid8mog objects that must be ne6sl557 (goal achieved)
        self.goal_wqid8mog = {
            get_parts(goal)[1]
            for goal in task.goals
            if get_parts(goal)[0] == "ne6sl557"
        }

        # Static adjacency chain for mz6t4 objects from lpmls56r facts: directed edges.
        self.adjacency = {}
        # Also add inverse edges to allow bidirectional searches.
        inverse_adj = {}

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "lpmls56r":
                frm, to = parts[1], parts[2]
                self.adjacency.setdefault(frm, set()).add(to)
                inverse_adj.setdefault(to, set()).add(frm)

        # Combine adjacency and inverse adjacency to allow bidirectional moves.
        for k, vs in inverse_adj.items():
            self.adjacency.setdefault(k, set()).update(vs)

        # Map from pt4b08k4 object to du770mtw objects linked by fqyrzl30 (?n54xx pt4b08k4, ?cz2xr du770mtw)
        self.fqyrzl30_links = {}
        for fact in task.initial_state | task.static:
            parts = get_parts(fact)
            if parts[0] == "fqyrzl30":
                pt4_obj, du_obj = parts[1], parts[2]
                self.fqyrzl30_links.setdefault(pt4_obj, set()).add(du_obj)

        # Collect all du770mtw objects that currently have ndg963qz predicate.
        self.ndg963qz_objects = {
            get_parts(fact)[1]
            for fact in task.initial_state
            if match(fact, "ndg963qz", "*")
        }

        # Pre-store static facts for adjacency for faster repeated search
        # Task static adjacency graph: used for shortest path distances between mz6t4 objects.

    def _find_shortest_distance(self, start, goal):
        """Find shortest distance in adjacency graph from start to goal (mz6t4 objects).
        If no path found, returns a large number (inf)."""
        if start == goal:
            return 0
        from collections import deque

        visited = set()
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            if current == goal:
                return dist
            visited.add(current)
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return float("inf")  # no path found

    def __call__(self, node):
        state = node.state

        # Goal achieved wqid8mog objects in current state
        achieved = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "ne6sl557", "*")
        }

        # Goal objects not yet achieved
        unachieved = self.goal_wqid8mog - achieved
        if not unachieved:
            # All goals met, heuristic is 0
            return 0

        # Current of6wn8m4 wqid8mog objects
        of6wn8m4_set = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "of6wn8m4", "*")
        }

        # Current ndg963qz du770mtw objects - resources needed by f7y6nzq3
        current_ndg963qz = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "ndg963qz", "*")
        }

        # Current qkdx0xe2 facts: map object -> mz6t4 location
        qkdx0xe2_map = {}
        for fact in state:
            if match(fact, "qkdx0xe2", "*", "*"):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                qkdx0xe2_map[obj] = loc

        # Approximate heuristic sum:
        total_cost = 0

        # For each unachieved wqid8mog object ?ris62:
        for ris62 in unachieved:
            # Step 1: Check if (of6wn8m4 ris62) holds; else account for 1 extra action needed to "refresh" it
            refresh_cost = 0 if ris62 in of6wn8m4_set else 1

            # Step 2: Find location (mz6t4) of ris62 via qkdx0xe2
            ris62_loc = qkdx0xe2_map.get(ris62, None)
            if ris62_loc is None:
                # If location unknown, count large penalty to avoid ignoring it
                ris62_loc = None

            # Step 3: For f7y6nzq3 to apply, need:
            # - some pt4b08k4 object n54xx with fqyrzl30(n54xx, cz2xr)
            # - with qkdx0xe2(n54xx, locX) where locX and ris62_loc are close (minimal qkdx0xe2 shifts)
            # - and that cz2xr has ndg963qz currently

            # Find candidate pt4b08k4 objects and their locations in state
            pt4_candidates = [
                (pt4_obj, qkdx0xe2_map.get(pt4_obj, None))
                for pt4_obj in self.fqyrzl30_links.keys()
                if qkdx0xe2_map.get(pt4_obj, None) is not None
            ]

            # Initialize minimal nm6lzp2x moves to large number
            best_shift_cost = float("inf")
            found_feasible = False

            for pt4_obj, pt4_loc in pt4_candidates:
                # For each du770mtw linked to pt4_obj
                for cz2xr in self.fqyrzl30_links[pt4_obj]:
                    if cz2xr in current_ndg963qz:
                        # Compute distance from pt4_loc to ris62_loc in adjacency graph of mz6t4
                        if ris62_loc is None or pt4_loc is None:
                            # Unknown location, skip this candidate
                            continue
                        dist = self._find_shortest_distance(pt4_loc, ris62_loc)
                        if dist == float("inf"):
                            continue
                        # Dist is number of nm6lzp2x (shift) actions needed (each moves qkdx0xe2 of pt4_obj once)

                        # The f7y6nzq3 action itself costs 1 action
                        # ndg963qz is removed after f7y6nzq3, so possibly 1 action to restore resource for next usage

                        # Keep minimal sum of costs for this goal object
                        candidate_cost = dist + 1 + 1  # shift actions + f7y6nzq3 + restoring resource (approximate)
                        if candidate_cost < best_shift_cost:
                            found_feasible = True
                            best_shift_cost = candidate_cost

            # If no feasible pt4b08k4 with ndg963qz found (resource exhausted), add penalty for refreshing or waiting
            if not found_feasible:
                # Penalize heavily because resource must be restored externally outside this heuristic model
                best_shift_cost = 5  # arbitrary penalty

            total_cost += refresh_cost + best_shift_cost

        return int(total_cost)
