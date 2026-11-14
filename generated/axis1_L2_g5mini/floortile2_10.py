from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return list of parts of a fact string '(pred a b ...)' -> ['pred','a','b',...]"""
    return fact[1:-1].split()


def match(fact, *pattern):
    """Return True if fact parts match the given pattern (wildcards '*' allowed)."""
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (hi4l5zmb).  The goal
    predicates in this domain are of the form (cb6fw2xd LOCATION TLF).
    Creating such a goal fact requires a single "assembly" action (kkqx0eg7 or
    wbsgr2yj) that consumes a particular resource (a qekv4465 object, called
    TLF here) and the presence of an en18p object at a neighboring location.
    The domain contains actions that:
      - move an en18p object between nh1sy8v9 locations along several relation
        types (vjcqgdgn, s4jrpk2b, vom4pfig, ga4l9do0);
      - change the qp2t6u4s mapping of an en18p to another TLF (wvswfzmn) if
        that target TLF exists as a fezojtj3 static fact;
      - create (cb6fw2xd LOCATION TLF) when a suitable en18p is located at a
        neighbor of LOCATION, the en18p currently has the required TLF via
        qp2t6u4s, and LOCATION currently has the fguw2142 marker.

    Assumptions
    -----------
    - We treat each goal (cb6fw2xd L T) independently and sum estimated costs
      because each assembly action produces one cb6fw2xd fact.
    - Movement of en18p objects through the nh1sy8v9 graph is possible along
      directed edges implied by static facts (a static fact "(REL A B)"
      allows movement from B -> A for the corresponding move action).
    - If an en18p does not currently hold a qp2t6u4s value, we conservatively
      assume it cannot be used to produce cb6fw2xd (no cheap way to create a
      qp mapping).
    - Repositioning the fguw2142 marker (required at the target LOCATION for
      the assembly action) is approximated optimistically: if fguw2142 is not
      already at LOCATION we add a small fixed cost (1) to reflect that the
      agent must be present there to execute the assembly. This keeps the
      heuristic simple and efficiently computable while still encouraging the
      search towards states where fguw2142 is in the right place.

    Heuristic Initialization
    ------------------------
    The constructor extracts the following from the planning task:
    - The set of goal facts (self.goals).
    - The set of TLF resources available statically (fezojtj3 facts).
    - A directed adjacency graph over nh1sy8v9 locations that encodes where an
      en18p can move in a single "move" action. For a static fact
          (REL A B)
      the adjacency edge is B -> A (because moves use the pattern (REL nb stq)).
      We include edges for REL in {vjcqgdgn, s4jrpk2b, vom4pfig, ga4l9do0}
      because all those relations permit movement actions.
    - For each LOCATION we precompute the set of predecessor locations S such
      that either "(s4jrpk2b LOCATION S)" or "(ga4l9do0 LOCATION S)" is true.
      These predecessors are the positions where an en18p must be located so
      that the assembly actions (wbsgr2yj or kkqx0eg7) can create
      (cb6fw2xd LOCATION T).
    All of the above are stored as efficient Python data structures so the
    heuristic evaluation is fast.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state the heuristic performs the following steps:

    1. Early exit: if all goal facts are already true in the state, return 0.

    2. Extract from the state:
       - which locations currently have fguw2142 (agent/marker positions),
       - the current location of each en18p via ypt9mrgu facts,
       - the qp2t6u4s mapping for each en18p, if present,
       - which cb6fw2xd goal facts are already satisfied.

    3. For every goal (cb6fw2xd LOC TLF) that is not yet true:
       a) If the goal is already present in the state, its cost is 0.
       b) Otherwise we need to produce it via an assembly action. We estimate
          the minimal cost to do so by considering each en18p object E that:
            - has a known current location (ypt9mrgu E L0),
            - has a qp2t6u4s mapping (an en18p without any mapping is ignored).
          For each such E we compute:
            i) distance = shortest number of move-steps (BFS in the directed
               adjacency graph) to bring E to any predecessor S of LOC where
               an assembly action can be applied (S satisfies (REL LOC S)
               with REL in {s4jrpk2b, ga4l9do0}). If no such reachable S,
               E cannot help with this goal.
           ii) change_cost = 0 if qp2t6u4s(E) == TLF else 1 if TLF is one of
               the available static fezojtj3 resources (we can then use
               wvswfzmn to change the mapping), otherwise E cannot produce
               the required TLF.
          The optimistic estimated cost for using E is:
              distance + change_cost + 1
          where +1 accounts for the assembly action (kkqx0eg7 or wbsgr2yj)
          that actually creates the cb6fw2xd fact.
       c) Agent presence cost: the assembly action requires fguw2142 at LOC.
          If fguw2142 is already in the state at LOC, agent_cost = 0,
          otherwise agent_cost = 1 (optimistic single action to position
          the agent; moving the agent is intertwined with moving en18p in the
          domain, but we approximate here for speed).
       d) For the goal we take the minimum total cost across all en18p candidates:
              best_goal_cost = min_over_E(distance + change_cost + 1 + agent_cost)
          If no en18p can produce the goal (no reachable candidate), return a
          large finite penalty value to denote the goal is currently hard /
          unreachable from this state.

    4. Return the sum of the best_goal_cost across all unsatisfied goals.
       This is a simple additive heuristic: each goal requires at least one
       assembly action that consumes resources and en18p movement; summing
       the independent estimates is an effective guidance for greedy
       best-first search in this domain.

    Notes on properties
    -------------------
    - The heuristic returns 0 iff all goals are satisfied (checked first).
    - It is efficiently computable: BFS runs on the small location graph,
      and we consider a small set of en18p candidates.
    - The heuristic is not guaranteed to be admissible (it is intentionally
      optimistic in places) but aims to be informative and fast to evaluate.
    """

    def __init__(self, task):
        # Store goals for quick checking.
        self.goals = task.goals
        static_facts = task.static

        # Resources (TLF values) that can be swapped to via wvswfzmn are encoded
        # by static (fezojtj3 TLF) facts.
        self.available_tlfs = {get_parts(f)[1] for f in static_facts if match(f, "fezojtj3", "*")}

        # Build directed adjacency graph for nh1sy8v9 locations.
        # For any static fact "(REL A B)" where REL in move_relations,
        # we add an edge B -> A (objects can be moved from B to A in one move).
        move_relations = {"vjcqgdgn", "s4jrpk2b", "vom4pfig", "ga4l9do0"}
        self.adj = defaultdict(set)  # node -> set(neighbor_nodes)
        # Also collect for assembly preconditions: predecessors of a target location
        # for REL in {s4jrpk2b, ga4l9do0}, because assembly actions require those.
        self.assembly_predecessors = defaultdict(set)  # target_loc -> set(predecessor_locs)

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in move_relations and len(parts) >= 3:
                A, B = parts[1], parts[2]
                # move from B -> A
                self.adj[B].add(A)
                # if this relation is one of the assembly relations, record predecessor
                if pred in {"s4jrpk2b", "ga4l9do0"}:
                    self.assembly_predecessors[A].add(B)

        # Keep the set of all nodes to help BFS initialization when needed.
        self.all_locations = set()
        for k, vs in self.adj.items():
            self.all_locations.add(k)
            self.all_locations.update(vs)
        # Ensure assembly_predecessors keys are in all_locations
        for k, vs in self.assembly_predecessors.items():
            self.all_locations.add(k)
            self.all_locations.update(vs)

    def shortest_distance_to_set(self, start, targets):
        """
        BFS on directed graph self.adj to compute shortest number of edges from
        start to any node in targets. Returns integer distance or None if no path.
        """
        if start in targets:
            return 0
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, d = q.popleft()
            for nbr in self.adj.get(node, ()):
                if nbr in visited:
                    continue
                if nbr in targets:
                    return d + 1
                visited.add(nbr)
                q.append((nbr, d + 1))
        return None

    def __call__(self, node):
        state = node.state

        # If already a goal state, heuristic is 0 (exact).
        if self.goals <= state:
            return 0

        # Extract current information from state.
        # fgu locations
        fgu_locations = {get_parts(f)[1] for f in state if match(f, "fguw2142", "*")}

        # ypt9mrgu: en18p -> location
        en_loc = {}
        for f in state:
            if match(f, "ypt9mrgu", "*", "*"):
                parts = get_parts(f)
                en, loc = parts[1], parts[2]
                en_loc[en] = loc

        # qp2t6u4s: en18p -> tlf
        en_tlf = {}
        for f in state:
            if match(f, "qp2t6u4s", "*", "*"):
                parts = get_parts(f)
                en, tlf = parts[1], parts[2]
                en_tlf[en] = tlf

        # For each goal (cb6fw2xd LOC TLF) compute optimistic minimal cost.
        total_cost = 0
        VERY_LARGE = 10 ** 6

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] != "cb6fw2xd" or len(parts) < 3:
                # This heuristic only reasons about cb6fw2xd goals. If other goals
                # exist, fall back to a small penalty per unsatisfied such goal.
                if goal not in state:
                    total_cost += 1
                continue

            _, loc, tlf = parts

            # If already true, skip.
            if goal in state:
                continue

            # The assembly actions need an en18p at a predecessor S of loc
            # where a static relation "(s4jrpk2b loc S)" or "(ga4l9do0 loc S)" holds.
            predecessors = self.assembly_predecessors.get(loc, set())
            if not predecessors:
                # No static relation allows assembling at loc; mark unreachable.
                return VERY_LARGE

            best_for_goal = None

            # Try all en18p candidates that have a known location and some qp mapping.
            for en, startloc in en_loc.items():
                # The assembly actions require qp2t6u4s(en, tlf) or a change to TLF
                # via wvswfzmn. wvswfzmn is applicable only if en currently has a qp mapping.
                if en not in en_tlf:
                    continue  # cannot change or be used for assembly (optimistic).

                # distance to any predecessor S
                dist = self.shortest_distance_to_set(startloc, predecessors)
                if dist is None:
                    continue

                # cost to change tlf if necessary
                change_cost = 0 if en_tlf.get(en) == tlf else (1 if tlf in self.available_tlfs else None)
                if change_cost is None:
                    continue

                # cost to ensure fguw2142 at loc (assembly precondition). Optimistic:
                agent_cost = 0 if loc in fgu_locations else 1

                # +1 for the assembly action itself
                candidate_cost = dist + change_cost + 1 + agent_cost

                if best_for_goal is None or candidate_cost < best_for_goal:
                    best_for_goal = candidate_cost

            if best_for_goal is None:
                # No candidate en18p can produce this goal now -> treat as very expensive.
                return VERY_LARGE

            total_cost += best_for_goal

        return total_cost