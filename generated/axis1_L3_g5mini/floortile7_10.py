from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the list of tokens for a fact string like '(pred a b)'."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the 'spanner' (secret_domain) domain.
    The domain's goals are facts of the form (tightened-2 <locatable> <location>).
    The most important dynamics for achieving such a goal are:
      - A 'link' predicate that indicates the current position of the
        "active" locatable (the one that can be used by actions producing
        tightened-2).
      - Operators that move the "link" (and the tightened-nut) along adjacency
        relationships between locatables (loose, at, usable, carrying).
      - A single action (walk / pickup_spanner) that produces the desired
        (tightened-2 <locatable> <location>) when a linked locatable is in the
        correct relation to the tightened nut and the target location.

    This heuristic therefore estimates how many link-moves are required to
    bring a linked locatable to each goal locatable and then adds one final
    action to generate the tightened-2 goal fact.

    Assumptions
    -----------
    - The (tightened-2 L Loc) goal for a given locatable L is created by an
      action (e.g. walk or pickup_spanner) that requires the acting locatable
      to currently have the predicate (link L') and reach the target L by
      moving along adjacency edges (loose, at, carrying, usable).
    - Moving the "link" one step along an adjacency edge corresponds to a
      single operator application that moves the link (and the tightened nut)
      from one locatable to an adjacent locatable.
    - Each missing (tightened-2 ...) goal can be satisfied independently; the
      heuristic sums a per-goal estimate. (This is not strictly accurate in
      presence of resource sharing, but is effective to guide greedy best-first.)
    - We consider adjacency edges to be undirected for the purpose of
      estimating move distances, because the domain contains several operators
      that move the tightened/link relation along different binary predicates.

    Heuristic Initialization
    ------------------------
    - Store the task goals so we can identify which facts are required.
    - Build an adjacency graph between locatable objects using static facts:
      any static fact with predicate in {loose, at, carrying, usable} that
      relates two locatables is interpreted as an undirected edge between them.
      This graph is used to estimate shortest move distances for the link.
    - Keep the set of goal facts for quick membership tests.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state):
    1. If all goal facts are already present in the state, return 0.
    2. Extract the set of currently linked locatables from the state (facts of
       the form "(link <locatable>)"). These are the starting points for moving
       the link.
    3. Build an adjacency graph that includes both static adjacency facts
       (collected at initialization) and adjacency facts that happen to be
       present in the current state (this makes the estimate robust if some
       adjacency is represented dynamically).
    4. For each goal of the form (tightened-2 <target_locatable> <location>):
       - If the goal fact is already true in the current state, cost = 0.
       - Otherwise compute the shortest number of adjacency steps from any
         currently linked locatable to the target locatable using a BFS on the
         adjacency graph (multi-source BFS). Each adjacency traversal is
         counted as one operator needed to move the link there.
       - Add +1 for the final operator that actually produces the
         (tightened-2 ...) fact once a linked locatable is adjacent/ready.
       - If the target is unreachable from any linked source in the adjacency
         graph, use a finite but large penalty to keep the heuristic finite.
    5. Sum the per-goal estimates and return the sum.

    Notes
    -----
    - The heuristic is not admissible (it intentionally simplfies interactions),
      but it is informative: it prefers states where linked locatables are
      closer to goal locatables.
    - The heuristic always returns finite values (uses a bounded penalty for
      unreachable targets) and returns 0 exactly for goal states.
    """

    def __init__(self, task):
        """
        Extract static adjacency information and goal facts.

        Parameters
        ----------
        task : Task
            The planning task object that provides static facts and goals.
        """
        self.goals = set(task.goals)
        self.static = set(task.static)

        # Build adjacency from static facts for predicates that indicate
        # a connection between two locatables.
        self.adj_static = {}
        adjacency_preds = {"loose", "at", "carrying", "usable"}

        for fact in self.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred in adjacency_preds and len(parts) >= 3:
                a, b = parts[1], parts[2]
                # treat as undirected edge
                self.adj_static.setdefault(a, set()).add(b)
                self.adj_static.setdefault(b, set()).add(a)

        # Precompute the set of goal target locatables (for quick filtering).
        self.goal_targets = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "tightened-2" and len(parts) >= 3:
                locatable = parts[1]
                location = parts[2]
                self.goal_targets.append((g, locatable, location))

    def __call__(self, node):
        """
        Compute the heuristic estimate for `node`.

        Parameters
        ----------
        node : SearchNode
            The search node, exposing `state` as a frozenset of fact strings.

        Returns
        -------
        int
            Estimated number of actions to reach the goal from the node's state.
        """
        state = node.state

        # If all goals satisfied -> heuristic 0
        if self.goals <= state:
            return 0

        # Build adjacency graph for this state (static adjacency plus any
        # adjacency facts present in the state).
        adj = {k: set(v) for k, v in self.adj_static.items()}
        adjacency_preds = {"loose", "at", "carrying", "usable"}
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred in adjacency_preds and len(parts) >= 3:
                a, b = parts[1], parts[2]
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)

        # Collect all currently linked locatables in the state.
        linked = {get_parts(f)[1] for f in state if get_parts(f)[0] == "link"}

        # If no linked locatable exists, we still want to return a finite
        # heuristic. Use tightened facts locations as plausible starting points
        # (a tightened nut at a locatable may be moved and create links).
        if not linked:
            linked = {get_parts(f)[2] for f in state if get_parts(f)[0] == "tightened"}
        # As a last resort, consider all locatables present in adjacency graph.
        if not linked:
            linked = set(adj.keys())

        # Helper: BFS multi-source to compute distance from any linked node to a target.
        def multi_source_bfs_distance(sources, target):
            if target in sources:
                return 0
            visited = set()
            q = deque()
            for s in sources:
                visited.add(s)
                q.append((s, 0))
            while q:
                v, d = q.popleft()
                for nb in adj.get(v, ()):
                    if nb in visited:
                        continue
                    if nb == target:
                        return d + 1
                    visited.add(nb)
                    q.append((nb, d + 1))
            # unreachable -> return a large finite penalty
            # Use graph size + 5 as penalty to remain finite and
            # scale with problem size.
            graph_size = max(1, len(adj))
            return graph_size + 5

        total = 0
        # For each tightened-2 goal, estimate cost if missing.
        for goal_fact, target_loc, target_locn in self.goal_targets:
            if goal_fact in state:
                continue
            dist = multi_source_bfs_distance(linked, target_loc)
            # +1 for the final action that produces (tightened-2 <target_loc> <location>)
            total += dist + 1

        # Ensure integer return
        return int(total)