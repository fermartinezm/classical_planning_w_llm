from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the tokens of a PDDL fact string, ignoring surrounding parentheses.

    Example:
      "(loose loc1 loc2)" -> ["loose", "loc1", "loc2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the "spanner" (secret_domain) domain.
    The heuristic estimates how many actions are needed to establish the missing
    tightened-2 facts required by the goal. It reasons about the nuts that can
    produce a given tightened-2 fact and about how far the current position
    of the nut's "tightened" marker is from a locatable that enables the
    final action which produces tightened-2 for the target locatable.

    Assumptions
    -----------
    - Each goal fact we care about is of the form (tightened-2 <locatable> <location>).
    - A tightened-2 fact for locatable L and location X can be produced by a
      single "final" action (e.g., "walk" / "pickup_spanner") if:
        * there exists a nut N with (loose-2 N X) in the current state, and
        * that nut N is currently tightened at some locatable S (a fact
          (tightened N S) exists in the state), and
        * the target locatable L is related by the static relation (loose L V)
          for some V and S can reach V by a series of moves that transfer the
          tightened marker along locatables.
    - The static structure (relations such as at, loose, carrying, usable)
      defines an adjacency graph that approximates possible single-step
      transfers of the tightened relation between locatables. This is an
      approximation that ignores resource contention on "link", but it gives
      a useful and efficient distance estimate.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - The set of goal facts (self.goals).
    - From the static facts we build:
      * an undirected adjacency graph between locatables using any of the
        relations that are used by transfer operators (at, loose, carrying,
        usable). This graph is used to estimate minimal number of transfers
        needed to move a tightened mark from one locatable to another.
      * a mapping self.loose_parents: for each locatable L, the set of locatables
        P such that (loose L P) is true in the static facts. These P nodes are
        the required positions of the tightened marker (before the final
        action) to enable an action that produces (tightened-2 L <location>).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goal facts are already true in the state, return 0.
    2. Extract the current nut-to-target-location mapping from the state's
       (loose-2 N X) facts (these can change during planning).
    3. Extract the current tightened positions of nuts from the state's
       (tightened N S) facts.
    4. For every goal (tightened-2 L X) that is NOT yet true in the state:
       a. Gather all nuts N for which (loose-2 N X) holds in the current state.
          These are the nuts that can produce a tightened-2 with location X.
       b. For each such nut N, find its current tightened position S (if any).
       c. The final action producing (tightened-2 L X) requires that the nut
          be tightened at some locatable V for which (loose L V) (these V are
          taken from the static loose relations). Compute the shortest path
          distance from S to any such V in the adjacency graph. Each edge in
          the adjacency graph corresponds to one transfer of the tightened
          marker (modeled as one domain action that moves the tightened
          relation). If S is already one of the required V's, the distance
          is zero.
       d. The estimated cost to achieve (tightened-2 L X) via nut N is:
            distance(S, V) + 1
          The +1 accounts for the final action that produces (tightened-2 L X).
       e. Take the minimum estimated cost across all candidate nuts N.
       f. If no candidate nut or path is found, use a finite fallback cost
          (a conservative bound) to ensure the heuristic remains finite.
    5. Sum the estimated costs for all missing tightened-2 goal facts and
       return that sum as the heuristic value.

    Notes on Design Choices
    -----------------------
    - The adjacency graph is an abstraction: it groups different domain
      relations (at, loose, usable, carrying) into a single undirected
      connectivity notion. This is intentional: operators that move the
      tightened relation use different domain predicates, but in most
      instances those predicates connect the same locatables in a
      chain-like structure. Using a single graph keeps the heuristic
      efficient while capturing the typical cost structure of the domain.
    - The heuristic is not admissible: it deliberately under/over-estimates
      in order to provide informative guidance for greedy best-first search.
    """

    def __init__(self, task):
        # The set of goal facts that must be true in a goal state.
        self.goals = task.goals

        # Extract static facts and build adjacency between locatables.
        static_facts = task.static

        # adjacency graph among locatables (undirected)
        self.adj = {}
        # mapping locatable -> set(locatable) where (loose locatable parent) is true
        self.loose_parents = {}

        def add_edge(a, b):
            if a not in self.adj:
                self.adj[a] = set()
            if b not in self.adj:
                self.adj[b] = set()
            self.adj[a].add(b)
            self.adj[b].add(a)

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            # Binary relations among locatables that we consider as adjacency.
            if pred in ("at", "loose", "carrying", "usable"):
                if len(parts) >= 3:
                    a, b = parts[1], parts[2]
                    add_edge(a, b)
                # For loose we also populate the parent mapping: (loose child parent)
                if pred == "loose" and len(parts) >= 3:
                    child, parent = parts[1], parts[2]
                    self.loose_parents.setdefault(child, set()).add(parent)

        # Ensure every locatable appears in adj dict (even if isolated).
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) >= 2:
                for tok in parts[1:]:
                    if tok not in self.adj:
                        self.adj.setdefault(tok, set())

    def _bfs_distance_to_targets(self, start, targets):
        """
        BFS to compute shortest distance from start to any node in targets
        in the adjacency graph self.adj. Returns integer distance or None if
        unreachable.
        """
        if start in targets:
            return 0
        if start not in self.adj:
            return None
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nb in self.adj.get(node, ()):
                if nb in visited:
                    continue
                if nb in targets:
                    return dist + 1
                visited.add(nb)
                queue.append((nb, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic estimate for the provided search node.

        Returns an integer estimate of the number of actions needed to reach
        a goal state from node.state.
        """
        state = node.state

        # 2) Heuristic must be zero for goal states.
        if self.goals <= state:
            return 0

        # Extract current nut -> target-location mapping from (loose-2 N X) facts in state.
        nut_to_target_location = {}
        # Extract current tightened positions for nuts from (tightened N S) facts in state.
        tightened_pos = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "loose-2" and len(parts) >= 3:
                nut, loc = parts[1], parts[2]
                nut_to_target_location.setdefault(loc, set()).add(nut)
            elif pred == "tightened" and len(parts) >= 3:
                nut, locatable = parts[1], parts[2]
                tightened_pos[nut] = locatable

        total_cost = 0
        # finite fallback per unreachable goal (keeps heuristic finite)
        FALLBACK_COST = 8

        # Consider every goal of type tightened-2 <locatable> <location>
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            pred = parts[0]
            if pred != "tightened-2" or len(parts) < 3:
                # ignore other goals (there shouldn't be others in this domain)
                continue
            locatable_goal, goal_location = parts[1], parts[2]
            goal_fact_str = f"(tightened-2 {locatable_goal} {goal_location})"
            # If the goal fact is already true, cost 0 for this goal.
            if goal_fact_str in state:
                continue

            # Candidate nuts that can produce a tightened-2 with goal_location.
            candidate_nuts = nut_to_target_location.get(goal_location, set())
            best_cost_for_goal = None

            # The final action that produces (tightened-2 locatable_goal goal_location)
            # requires the nut to be tightened at some locatable V where (loose locatable_goal V)
            required_parents = self.loose_parents.get(locatable_goal, set())

            if not required_parents:
                # If static loose relation does not define any parent for the goal locatable,
                # it's unlikely that the standard final operators can create tightened-2
                # for this locatable. Fall back to a cost but still finite.
                best_cost_for_goal = FALLBACK_COST

            else:
                for nut in candidate_nuts:
                    start_loc = tightened_pos.get(nut)
                    if start_loc is None:
                        # If we don't know where the nut is tightened currently, assume an extra cost.
                        # Still consider a fallback but prefer other nuts.
                        continue
                    dist = self._bfs_distance_to_targets(start_loc, required_parents)
                    if dist is not None:
                        # +1 for the final action that produces tightened-2
                        cost = dist + 1
                        if best_cost_for_goal is None or cost < best_cost_for_goal:
                            best_cost_for_goal = cost

            if best_cost_for_goal is None:
                # No viable candidate nut found or no path -> use fallback finite cost.
                best_cost_for_goal = FALLBACK_COST

            total_cost += best_cost_for_goal

        return int(total_cost)