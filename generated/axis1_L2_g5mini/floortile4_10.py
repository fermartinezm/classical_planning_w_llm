from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into tokens."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain ("hi4l5zmb") domain.  The
    main goal in this domain is to create facts of the form
    (cb6fw2xd <location> <tlfqu>).  Producing such a fact requires:
      - an en18p object (type k9b7lknr) that carries or is associated with the
        target tlfqu via (qp2t6u4s <en18p> <tlfqu>) (or by changing this mapping
        via the wvswfzmn action),
      - the en18p to be located at a node stq0j that has a relation to the
        destination nb9v4 (one of s4jrpk2b / vjcqgdgn / vom4pfig / ga4l9do0
        appears as "(pred nb9v4 stq0j)"),
      - the destination nb9v4 must have (fguw2142 nb9v4) true when the placing
        operator (kkqx0eg7 or wbsgr2yj) is applied.  The placing operator
        consumes that fguw2142 fact.
    The heuristic estimates the number of actions required by treating the
    problem roughly as:
      (cost to ensure an en18p is bound to the required tlfqu)
      + (cost to move that en18p to a suitable adjacent stq0j)
      + (cost to ensure the destination has fguw2142)
      + (1 action to produce the cb6fw2xd fact).
    The heuristic sums the estimate for each unmet cb6fw2xd goal.  It uses
    simple BFS distances on a graph induced by the static adjacency relations
    to estimate required move actions.

    Assumptions
    -----------
    - The static relations s4jrpk2b, vjcqgdgn, vom4pfig and ga4l9do0 describe
      possible local connections between nh1sy8v9 nodes.  If a static fact
      is "(pred A B)" for any of those preds, an en18p can move from B to A
      in one action (this matches how the domain actions require a relation
      of the form "(pred nb9v4 stq0j)").
    - Changing an en18p's qp2t6u4s mapping to the required tlfqu costs 1
      action (wvswfzmn).
    - Moving an en18p along one adjacency step costs 1 action (h5dglxse,
      ck3rwuwy, a5xd8fg8 or an2ipri5).
    - Bringing a free slot (fguw2142) to a particular node is approximated
      by the undirected shortest-path distance between a node that currently
      has fguw2142 and the desired node.
    - Interactions and overlaps (for example an en18p's movements simultaneously
      fixing both its position and the location of a free slot) are not
      precisely modeled; the heuristic sums independent estimates, making it
      generally informative but not guaranteed admissible.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
      - self.goals_cb: the set of goal facts that are "cb6fw2xd" (these are
        the goal facts we specifically estimate for).
      - static adjacency information: builds two structures:
          * self.incoming_sources[dest] = {src nodes} where a relation
            of the form "(pred dest src)" exists for any of the four movement
            relation predicates.  An en18p must be at one of those src nodes
            to perform the placement to dest.
          * self.undirected_neighbors[node] = set(neighbor nodes) used to
            compute undirected distances (for approximating movement of
            fguw2142 slots).
    """

    def __init__(self, task):
        """Extract goals and static adjacency information from the task."""
        # Collect cb6fw2xd goals only (these are the facts we estimate).
        self.goals_cb = {g for g in task.goals if get_parts(g)[0] == "cb6fw2xd"}

        # Build adjacency structures from relevant static relations.
        static = task.static

        # Relations that allow en18p to move from a source node to a destination
        # if static contains "(pred dest source)". We'll treat them uniformly.
        move_preds = {"s4jrpk2b", "vjcqgdgn", "vom4pfig", "ga4l9do0"}

        # incoming_sources[dest] = set of source nodes stq0j such that
        # relation "(pred dest stq0j)" exists in static. In domain, to place
        # at dest nb9v4 we need an en18p at such a stq0j.
        self.incoming_sources = {}

        # undirected neighbor map used to compute undirected shortest paths.
        self.undirected_neighbors = {}

        # Also collect set of all nh1sy8v9 nodes known from static.
        for fact in static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in move_preds and len(parts) >= 3:
                dest, src = parts[1], parts[2]
                self.incoming_sources.setdefault(dest, set()).add(src)
                # Undirected neighbors for both nodes
                self.undirected_neighbors.setdefault(dest, set()).add(src)
                self.undirected_neighbors.setdefault(src, set()).add(dest)

        # Ensure all nodes appear in undirected map (even if isolated)
        for fact in static:
            parts = get_parts(fact)
            for token in parts[1:]:
                self.undirected_neighbors.setdefault(token, set())

    def _bfs_directed_to_targets(self, start, targets):
        """
        BFS on the directed movement graph (edges: node -> neighbor) where an
        en18p can move from its current node to neighbors constructed from the
        static incoming_sources mapping. Return shortest distance (in moves)
        from start to any node in targets. If unreachable, return None.
        """
        if start in targets:
            return 0
        # Build directed adjacency from incoming_sources: if incoming_sources[d]
        # contains s, then there is a directed edge s -> d (en moves from s to d).
        # Precompute neighbors lazily each call to avoid storing global directed map.
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            # Expand neighbors: all destinations d such that node is a source for d
            # i.e., node in incoming_sources.get(d)
            for dest, sources in self.incoming_sources.items():
                if node in sources:
                    if dest in targets:
                        return dist + 1
                    if dest not in visited:
                        visited.add(dest)
                        q.append((dest, dist + 1))
        return None

    def _bfs_undirected_distance(self, starts, target):
        """
        Standard undirected BFS from any node in 'starts' to 'target' using
        self.undirected_neighbors. Returns distance (#edges) or None if not
        reachable.
        """
        if target in starts:
            return 0
        visited = set(starts)
        q = deque([(s, 0) for s in starts])
        while q:
            node, dist = q.popleft()
            for nbr in self.undirected_neighbors.get(node, ()):
                if nbr == target:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    q.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute heuristic estimate for a given search node.

        Steps:
        1. If all cb6fw2xd goals already satisfied in node.state, return 0.
        2. Parse current state to discover:
           - where en18p objects are: (ypt9mrgu <en> <loc>)
           - which en18p objects have which tlfqu mapping: (qp2t6u4s <en> <tlfqu>)
           - which nodes currently have fguw2142 (free slots)
        3. For every cb6fw2xd goal not yet satisfied:
           a) cost_assign = 0 if some en already has qp2t6u4s <tlfqu>, else 1
              (one wvswfzmn to change some en's mapping).
           b) cost_move_en = shortest directed distance (#moves) for an en that
              will perform the placement to reach any source stq0j such that
              there is a static relation "(pred goal_loc stq0j)". If no en with
              the correct tlf exists, any en may be chosen after assignment.
           c) cost_prepare_free = 0 if (fguw2142 goal_loc) currently true;
              otherwise approximate the steps needed to bring a free slot to
              goal_loc by the undirected shortest-path distance between any
              node that currently has fguw2142 and goal_loc.
           d) final action cost = 1 for placing (kkqx0eg7/wbsgr2yj).
           e) The goal's estimated cost is cost_assign + cost_move_en +
              cost_prepare_free + 1. If any component is unreachable we
              assign a large finite penalty (1000) so the heuristic remains
              finite.
        4. Return the sum of estimated costs for all unmet cb6fw2xd goals.
        """
        state = node.state

        # Quick goal check: if the complete goal set is satisfied, heuristic is 0.
        if self.goals_cb <= state:
            return 0

        # Parse dynamic facts in the state.
        # Maps en -> location, en -> tlfqu, and set of nodes with fguw2142.
        en_location = {}   # ypt9mrgu
        en_tlf = {}        # qp2t6u4s
        fguw_nodes = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "ypt9mrgu" and len(parts) >= 3:
                en, loc = parts[1], parts[2]
                en_location[en] = loc
            elif pred == "qp2t6u4s" and len(parts) >= 3:
                en, tlf = parts[1], parts[2]
                en_tlf[en] = tlf
            elif pred == "fguw2142" and len(parts) >= 2:
                fguw_nodes.add(parts[1])

        all_en = set(en_location.keys()) | set(en_tlf.keys())
        if not all_en:
            # No en objects found in the state; be conservative but finite.
            # This is unlikely for proper instances but keeps heuristic finite.
            return 1000

        total_cost = 0
        LARGE = 1000

        # For each required cb6fw2xd goal, compute an estimate if unmet.
        for goal in self.goals_cb:
            if goal in state:
                continue  # Already satisfied.
            parts = get_parts(goal)
            # Format: (cb6fw2xd <location> <tlfqu>)
            if len(parts) < 3:
                # malformed goal; skip with penalty
                total_cost += LARGE
                continue
            _, goal_loc, goal_tlf = parts[0], parts[1], parts[2]

            # (a) assignment cost: 0 if some en already has the desired tlfqu.
            ens_with_tlf = [en for en, tlf in en_tlf.items() if tlf == goal_tlf]
            cost_assign = 0 if ens_with_tlf else 1

            # (b) movement cost for an en to reach some source stq0j where
            # placement operator can be applied. We need incoming_sources[goal_loc].
            sources = self.incoming_sources.get(goal_loc, set())
            if not sources:
                # No incoming source nodes for this goal location: unreachable
                cost_move_en = None
            else:
                # If there are en that already have the right tlf, prefer them.
                candidate_ens = ens_with_tlf if ens_with_tlf else list(all_en)
                move_dists = []
                for en in candidate_ens:
                    start = en_location.get(en)
                    if start is None:
                        # If we don't know an en's position, pessimistically assume large
                        continue
                    dist = self._bfs_directed_to_targets(start, sources)
                    if dist is not None:
                        move_dists.append(dist)
                if move_dists:
                    cost_move_en = min(move_dists)
                else:
                    # No reachable en to a valid source stq0j found.
                    cost_move_en = None

            # If no en currently has tlf but assignment allowed, try to estimate
            # using any en (we already used any en if ens_with_tlf empty above).
            if cost_move_en is None and cost_assign == 1:
                # Try again with any en (we might have skipped some earlier because
                # en_location missing); pick best known en.
                move_dists_any = []
                for en in all_en:
                    start = en_location.get(en)
                    if start is None:
                        continue
                    dist = self._bfs_directed_to_targets(start, sources)
                    if dist is not None:
                        move_dists_any.append(dist)
                if move_dists_any:
                    cost_move_en = min(move_dists_any)
                else:
                    cost_move_en = None

            # (c) cost to prepare fguw2142 at goal_loc.
            if goal_loc in fguw_nodes:
                cost_prepare_free = 0
            else:
                if not fguw_nodes:
                    cost_prepare_free = None
                else:
                    d = self._bfs_undirected_distance(fguw_nodes, goal_loc)
                    cost_prepare_free = d if d is not None else None

            # If any component is unreachable mark this goal with LARGE cost.
            if cost_move_en is None or cost_prepare_free is None:
                total_cost += LARGE
                continue

            # Final action to generate cb6fw2xd (kkqx0eg7 or wbsgr2yj) costs 1.
            goal_cost = cost_assign + cost_move_en + cost_prepare_free + 1
            total_cost += goal_cost

        return total_cost