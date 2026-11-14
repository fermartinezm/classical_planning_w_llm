from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards.

    :param fact: The fact as a string, e.g. "(predicate obj1 obj2)"
    :param args: Sequence of strings or wildcards "*" to match against fact parts
    :return: True if the fact matches the pattern, False otherwise
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (hi4l5zmb).

    Summary
    -------
    Estimates the number of actions required to reach the goal state by focusing on
    the goal facts of type `(cb6fw2xd ?location ?qekv4465)`, which appear to represent
    the main goals (based on provided instances). The heuristic measures how many of
    these goal facts are currently missing and estimates how many actions are needed
    to produce them, considering the domain dynamics of moving the `ypt9mrgu` relations
    (seemingly "possession" or "assignment" of k9b7lknr objects to nh1sy8v9 objects)
    and the fluent `(fguw2142 ?nh1sy8v9)`, which likely represent movable tokens or supports.

    Assumptions
    -----------
    - Goals only require the predicates `(cb6fw2xd location qekv4465)` to hold.
    - Each goal `(cb6fw2xd L Q)` must be achieved by transferring the relationship `(qp2t6u4s k9b7lknr Q)`
      together with the correct positioning of related objects, involving `ypt9mrgu` and `fguw2142`.
    - Achieving each `(cb6fw2xd ...)` goal fact roughly corresponds to one complex move sequence.
    - Static facts encode the graph-like relations `(s4jrpk2b)`, `(vom4pfig)`, `(vjcqgdgn)`, `(ga4l9do0)`,
      representing connectivity or reachability among objects.
    - The heuristic does not need to be admissible but should guide the search efficiently.

    Heuristic Initialization
    ------------------------
    - Extract all goal `(cb6fw2xd ?loc ?qek)` facts as the core goals.
    - Extract from static facts the connectivity predicates `(s4jrpk2b)`, `(vom4pfig)`, `(vjcqgdgn)`, `(ga4l9do0)`
      building them into adjacency mappings to quickly estimate reachability and steps.
    - Extract the set of all objects involved: k9b7lknr (`k_objs`), nh1sy8v9 (`n_objs`), and qekv4465 (`q_objs`).
    - Extract from static facts the graphs for movement, e.g., the neighborhood relationships among nh1sy8v9 objects.
    - Precompute shortest path distances between nh1sy8v9 objects using the static adjacency info for estimating move costs.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify the set of goal pairs `(location, qekv)` from goal facts `(cb6fw2xd ...)`.
    2. For each goal pair, check if the predicate currently holds in the state:
       - If yes, no cost is added.
       - If not, estimate how many actions are needed to produce it.

    3. To produce `(cb6fw2xd L Q)`:
       - We must have `(qp2t6u4s E Q)` for some `E` of type k9b7lknr.
       - We need to have `(ypt9mrgu E N)` where `N` is an nh1sy8v9 reachable/configured to allow switching `(ypt9mrgu E L)`.
       - There is a corresponding `(fguw2142 N)` fluent acting like a “position marker.”
       - Creating `(cb6fw2xd L Q)` requires moving the position marker(s) and swapping the `ypt9mrgu` relations appropriately.

    4. Use shortest path distances in the adjacency graph of nh1sy8v9 objects (built from predicates `(s4jrpk2b)`, `(vom4pfig)`, `(vjcqgdgn)`, `(ga4l9do0)`) to estimate moves between positions where `(fguw2142)` holds and target locations `L`.
    5. Summing minimal movement costs to move the marker and do relevant `ypt9mrgu` transfers for each missing goal fact.
    6. Add a small fixed cost per goal fact for the required predicate switches.
    7. Return the sum over all goal pairs of these estimates.

    Notes:
    - The heuristic runs efficiently by caching adjacency maps and shortest paths on initialization.
    - It returns 0 if all goal `(cb6fw2xd ...)` facts hold.
    - It ensures a finite cost for reachable states.
    """

    def __init__(self, task):
        self.goals = set(task.goals)
        self.static = set(task.static)

        # Extract all goal predicates of type (cb6fw2xd L Q)
        self.goal_pairs = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "cb6fw2xd" and len(parts) == 3:
                self.goal_pairs.add((parts[1], parts[2]))

        # Extract static adjacency relations among nh1sy8v9 objects
        # We'll build a graph: nh1sy8v9 -> set of neighbors, merging all 4 connectivity preds
        self.adjacency = {}
        self.nh_objs = set()
        predicates_to_collect = ["s4jrpk2b", "vom4pfig", "vjcqgdgn", "ga4l9do0"]

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] in predicates_to_collect and len(parts) == 3:
                src, dst = parts[1], parts[2]
                self.nh_objs.add(src)
                self.nh_objs.add(dst)
                if src not in self.adjacency:
                    self.adjacency[src] = set()
                self.adjacency[src].add(dst)
                # Relations appear directed; to be safe, add reverse if same predicate appears reversed
                # We do not assume symmetry - so why not keep as directed to reduce complexity
                
        # Some objects from domain types might be missing in adjacency — ensure they have empty adjacency
        for o in self.nh_objs:
            if o not in self.adjacency:
                self.adjacency[o] = set()

        # Precompute shortest paths between all nh1sy8v9 objects using BFS for efficiency
        self.shortest_paths = {}
        for obj in self.nh_objs:
            self.shortest_paths[obj] = self._bfs_distances(obj)

        # Extract set of k9b7lknr and qekv4465 objects from static facts (for validation/usage)
        # We will not use full sets since domain is complex, but might be handy to know k9b7lknr related to goals
        self.k_objs = set()
        self.q_objs = set()
        for fact in self.static:
            parts = get_parts(fact)
            # Collect all k9b7lknr in ypt9mrgu and qp2t6u4s predicates
            if parts[0] == "ypt9mrgu" and len(parts) == 3:
                self.k_objs.add(parts[1])
            if parts[0] == "qp2t6u4s" and len(parts) == 3:
                self.k_objs.add(parts[1])
                self.q_objs.add(parts[2])

    def _bfs_distances(self, start):
        """Compute shortest distances from start to all reachable nh1sy8v9 nodes."""
        distances = {start: 0}
        frontier = [start]
        while frontier:
            current = frontier.pop(0)
            for neighbor in self.adjacency.get(current, ()):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    frontier.append(neighbor)
        return distances

    def __call__(self, node):
        state = node.state

        # Helper: Does the goal cb6fw2xd hold?
        def has_goal_cb6fw2xd(loc, q):
            return f"(cb6fw2xd {loc} {q})" in state

        # Helper: Facts by predicate name. We'll build fast lookup dictionaries.
        facts_by_pred = {}
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            facts_by_pred.setdefault(pred, set()).add(fact)

        # Extract current ypt9mrgu relations (E,N)
        current_ypt9mrgu = set()
        for fact in facts_by_pred.get("ypt9mrgu", ()):
            parts = get_parts(fact)
            if len(parts) == 3:
                current_ypt9mrgu.add((parts[1], parts[2]))  # (k9b7lknr, nh1sy8v9)

        # Extract current fguw2142 fluents (positions of markers)
        current_fguw2142 = set()
        for fact in facts_by_pred.get("fguw2142", ()):
            parts = get_parts(fact)
            if len(parts) == 2:
                current_fguw2142.add(parts[1])  # nh1sy8v9 location holding marker

        # Extract qp2t6u4s facts (relation between k9b7lknr and qekv4465)
        current_qp2t6u4s = {}
        for fact in facts_by_pred.get("qp2t6u4s", ()):
            parts = get_parts(fact)
            if len(parts) == 3:
                kobj, qobj = parts[1], parts[2]
                current_qp2t6u4s[qobj] = kobj

        heuristic_value = 0

        for (goal_loc, goal_q) in self.goal_pairs:
            if has_goal_cb6fw2xd(goal_loc, goal_q):
                # This goal achieved; no cost.
                continue

            # Goal not achieved: estimate effort
            # Step 1: find k9b7lknr E for this qekv4465 goal_q via qp2t6u4s
            if goal_q not in current_qp2t6u4s:
                # If no such k-object assigned, assume minimal fixed cost for assignment
                # Here, assign cost 3 as an arbitrary cost for this missing binding
                heuristic_value += 3
                continue

            kobj = current_qp2t6u4s[goal_q]

            # Step 2: we want to have ypt9mrgu(kobj, N) where N is an nh1sy8v9 object
            # that can be moved to goal_loc to create cb6fw2xd(goal_loc, goal_q).

            # Find current location N0 of kobj from ypt9mrgu:
            locs_of_kobj = [n for (k, n) in current_ypt9mrgu if k == kobj]
            if not locs_of_kobj:
                # kobj has no ypt9mrgu assigned, assume minimal cost 3 for setup
                heuristic_value += 3
                continue
            current_loc = locs_of_kobj[0]

            # Step 3: Need to move marker (fguw2142) from its current location to goal_loc:
            # Find distance from marker position(s) to goal_loc

            # If marker is currently at current_loc or nearby? The exact mechanism is complex,
            # so approximate by choosing closest marker position to goal_loc and distance from current_loc
            min_marker_dist = None
            for marker_pos in current_fguw2142:
                dist = self.shortest_paths.get(marker_pos, {}).get(goal_loc, float('inf'))
                if min_marker_dist is None or dist < min_marker_dist:
                    min_marker_dist = dist
            if min_marker_dist is None:
                min_marker_dist = 3  # fallback arbitrary cost

            # Step 4: Distance between current_loc of kobj and goal_loc (where cb6fw2xd must hold)
            dist_ypt = self.shortest_paths.get(current_loc, {}).get(goal_loc, float('inf'))
            if dist_ypt == float('inf'):
                dist_ypt = 4  # fallback arbitrarily large cost but finite

            # Approximate number of moves needed to update relations:
            # Assume cost proportional to moving the marker + moving the ypt9mrgu relation
            # Plus constant for the swap action(s) that produce the goal predicate.

            moves_cost = min_marker_dist + dist_ypt + 2  # 2 actions for switching ypt9mrgu and updating fguw2142

            heuristic_value += moves_cost

        # Zero if goals achieved, finite otherwise
        return heuristic_value
