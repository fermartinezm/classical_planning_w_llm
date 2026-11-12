from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a PDDL fact string by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (u2gh8ts5).  The domain's goals
    are facts of the form (wuqjn5r5 <kuk> <z1>), i.e. assign a kuk-object to a
    particular z1-node.  The only operator that directly changes these facts is
    `yaz3jy7e`, which effectively moves a wuqjn5r5 relation from one z1-node to
    a successor z1-node.  Therefore this heuristic estimates the number of
    actions required to achieve each desired (wuqjn5r5 kuk target) by computing
    a shortest-path distance in the static directed graph induced by the
    k21e3pdq facts from the kuk's current z1-node to the goal z1-node and
    summing those distances for all goal facts not yet satisfied.

    Assumptions
    -----------
    - Moving a wuqjn5r5 relation along a single k21e3pdq edge can be achieved
      in one effective step (corresponding to one application of the domain's
      action that transfers a wuqjn5r5 along one graph edge). This ignores
      some detailed preconditions (odqd6scj, ajcpzwcf toggles) but produces a
      fast informative estimate for greedy best-first search.
    - If a particular kuk has multiple wuqjn5r5 facts in the state, the closest
      one (in terms of static graph distance) is assumed to be usable.
    - If a kuk does not occur in any wuqjn5r5 fact in the current state, the
      heuristic treats the goal as currently unreachable by assigning a large
      finite cost (rather than infinity), allowing the planner to continue
      search but disfavoring such branches.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts static information from the task:
    - All k21e3pdq facts are used to build a directed adjacency map between z1
      nodes. Each fact (k21e3pdq A B label) produces an edge A -> B (label is
      ignored for path length computation).
    - The task goals are recorded so the heuristic can quickly check for goal
      satisfaction and determine which (wuqjn5r5 ...) facts are required.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state):
    1. If all goal facts are present in the state's fact set, return 0.
    2. Parse the state's wuqjn5r5 facts into a map: kuk -> list of current z1 nodes
       where that kuk is currently attached. (Multiple entries are allowed;
       the best one will be selected.)
    3. For each goal fact of the form (wuqjn5r5 kuk target):
       a. If the exact goal fact is already true in the state, cost contribution
          is 0.
       b. Otherwise, find the current positions (start nodes) for this kuk from
          the map created in step 2. If no such start exists, assign a large
          finite cost (a tunable penalizing constant) for this goal.
       c. Run a BFS on the static adjacency graph from any start node to the
          goal node to compute the shortest number of directed edges (each
          representing a single expected wuqjn5r5 transfer). If the target is
          reachable, that shortest-path length is used as this goal's cost.
          If unreachable, again assign the large finite penalty.
    4. Sum the per-goal costs and return the total as the heuristic value.

    Notes
    -----
    - The heuristic is intentionally optimistic regarding auxiliary preconditions
      needed by the real domain operators (odqd6scj, ajcpzwcf manipulation).
      This makes it fast and informative for guiding greedy search while not
      being admissible.
    - The penalty for missing start positions or unreachable goals is finite
      (so the heuristic always returns a finite number). The constant used for
      the penalty can be tuned; it is chosen here to be large enough to penalize
      impossible-looking configurations relative to normal path lengths.
    """

    def __init__(self, task):
        """Extract static graph structure and store goals."""
        self.goals = task.goals  # set/frozenset of goal fact strings
        static_facts = task.static

        # Build adjacency list from static k21e3pdq facts.
        # Each fact has form "(k21e3pdq src dst label)" -> create edge src -> dst.
        self.adj = {}  # maps src -> set of dst
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "k21e3pdq" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.adj.setdefault(src, set()).add(dst)

        # Precompute set of all nodes (sources and destinations).
        nodes = set(self.adj.keys())
        for dsts in self.adj.values():
            nodes.update(dsts)
        self.nodes = nodes

        # A reasonably large finite penalty for "no known source" or unreachable.
        # This keeps heuristic finite while strongly penalizing such goals.
        self.unreachable_penalty = max(50, len(self.nodes) + 10)

    def _shortest_distance(self, starts, target):
        """
        Compute the shortest directed distance from any node in `starts` to `target`
        in the static adjacency graph using BFS. Returns an integer distance or None
        if the target is not reachable.
        """
        if target in starts:
            return 0
        visited = set()
        q = deque()
        for s in starts:
            if s in self.nodes:
                q.append((s, 0))
                visited.add(s)
        while q:
            node, dist = q.popleft()
            for nbr in self.adj.get(node, ()):
                if nbr == target:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    q.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic estimate for `node`.

        Returns the sum of shortest-path distances (in the static k21e3pdq graph)
        from each kuk's current z1-node(s) to the goal z1-node required by each
        (wuqjn5r5 kuk target) goal that is not yet satisfied. If a kuk has no
        current wuqjn5r5 occurrences in the state or the goal is unreachable in
        the static graph, a large finite penalty is used for that goal.
        """
        state = node.state

        # If the node already satisfies all goals, heuristic is zero.
        if self.goals <= state:
            return 0

        # Map kuk -> set of current z1 nodes where (wuqjn5r5 kuk z1) holds.
        kuk_positions = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "wuqjn5r5" and len(parts) >= 3:
                kuk, znode = parts[1], parts[2]
                kuk_positions.setdefault(kuk, set()).add(znode)

        total_cost = 0

        # Consider only goal facts that are wuqjn5r5 (others ignored, but domain
        # uses wuqjn5r5 goals).
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            if parts[0] != "wuqjn5r5" or len(parts) < 3:
                # Not a predicate this heuristic reasons about; ignore.
                continue
            kuk, target = parts[1], parts[2]
            # If already satisfied in the state, no cost.
            if goal in state:
                continue

            starts = kuk_positions.get(kuk)
            if not starts:
                # No current position known for this kuk -> penalize.
                total_cost += self.unreachable_penalty
                continue

            dist = self._shortest_distance(starts, target)
            if dist is None:
                total_cost += self.unreachable_penalty
            else:
                # Each edge corresponds to one effective wuqjn5r5-transfer action.
                total_cost += dist

        return total_cost