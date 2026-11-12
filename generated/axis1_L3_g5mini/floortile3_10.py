from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract predicate and arguments from a PDDL fact string '(pred a b ...)'."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the "spanner" (secret_domain) planning domain.
    It estimates the number of actions required to achieve all goals of the
    form (tightened-2 <locatable> <location>) from a given state.

    Assumptions
    -----------
    - Goals are only of the form (tightened-2 <locatable> <location>).
    - The primary mechanism to achieve a goal (tightened-2 L loc) is:
        1) have a nut that is tightened to some locatable node b which is a
           predecessor of the goal locatable L (according to static relations
           such as loose/at/carrying/usable where (pred L b) exists),
        2) ensure the nut's loose-2 fact identifies the goal location `loc`
           (if not, one action can move the nut's loose-2 to the desired
           location via tighten_nut),
        3) execute the final operator (walk or pickup_spanner) that produces
           (tightened-2 L loc).
    - Moving the tightened attachment from one locatable to an adjacent
      locatable is modeled by traversing an edge in a directed graph derived
      from static facts. Each graph edge corresponds roughly to one operator
      application that moves a tightened relation along the chain.
    - We ignore transient constraints such as availability/consumption of
      `(link ...)` facts and operator interference; this makes the heuristic
      inadmissible but computationally cheap and informative for guiding
      greedy best-first search.

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - store the goal set (self.goals),
    - build a directed adjacency graph among locatable nodes from static facts.
      An adjacency edge u -> v exists if there is a static fact like
      (loose v u), (at v u), (carrying v u) or (usable v u). This models the
      fact that a tightened relation attached to u can be moved to v in one
      operator application (ignoring temporary `link` constraints).
    - also build a "predecessors" mapping: predecessors[v] is the set of nodes
      u such that there is an adjacency edge u -> v (these u are candidate
      places where a nut has to be tightened immediately before the final
      action creating (tightened-2 v loc)).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node / state:
    1. If all goals are already present in the state -> heuristic = 0.
    2. Parse the current state's relevant facts:
       - For each nut, record the locatable it is currently tightened to via
         facts of the form "(tightened nutX locY)".
       - For each nut, record the location it is currently loose at via
         facts "(loose-2 nutX locationZ)".
    3. For each goal (tightened-2 G LOC):
       a) If the exact fact is in the state, cost contribution is 0.
       b) Otherwise, consider every nut `n` that has a known tightened
          locatable start_loc = tightened(n):
            i) Compute shortest-path distances from start_loc to all
               locatables using the adjacency graph built from static facts.
           ii) Identify the set of candidate predecessor nodes P = predecessors[G]
               (these are nodes `b` such that an operator exists that can turn
               a tightened-at-`b` nut into tightened-2 at G).
          iii) For this nut `n` the minimal work to make (tightened-2 G LOC)
               is:
                 dist(start_loc, b)   # number of "move tightened" steps to b
                 + (0 if loose-2(n) == LOC else 1)  # optionally fix nut's loose-2
                 + 1  # the final operator (walk/pickup_spanner) producing tightened-2
               where b ranges over P and we pick the minimal dist.
       c) The estimated cost for that goal is the minimal value found across
          all nuts. If no nut or path yields a finite value, we assign a
          finite penalty (large constant) to keep the heuristic finite.
    4. The heuristic is the sum of the per-goal estimates. This assumes goals
       are mostly independent; while this may overcount shared opportunities,
       it gives a simple, informative and fast-to-compute estimate.

    Notes
    -----
    - The heuristic is intentionally optimistic about resource contention
      (e.g., repeated use of `link`), which makes it inadmissible but
      effective for guiding greedy search.
    - All parsing of facts strips surrounding parentheses and splits by space.
    """

    def __init__(self, task):
        """
        Build static adjacency and predecessor graphs from static facts.

        Params:
        - task: a Task instance (see planner code). We extract:
            self.goals = task.goals
            static_facts = task.static
        """
        self.goals = task.goals
        static_facts = task.static

        # adjacency[u] = set(v) means tightened can move from u to v in one operator
        self.adjacency = defaultdict(set)
        # predecessors[v] = set(u) means there's an edge u -> v (u is a candidate
        # tightened location just before producing tightened-2 at v).
        self.predecessors = defaultdict(set)

        # Predicates that allow moving tightened from a node to its neighbor.
        move_preds = {"loose", "at", "carrying", "usable"}

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            # only consider binary relations locatable1 locatable2
            if pred in move_preds and len(parts) == 3:
                a, b = parts[1], parts[2]
                # An operator moves tightened from b -> a (if tightened is at b it can
                # be moved to a in one step when other preconditions hold).
                self.adjacency[b].add(a)
                self.predecessors[a].add(b)

        # Collect set of locatable nodes known from static info (helpful for BFS)
        self.all_locatables = set(self.adjacency.keys())
        for targets in self.adjacency.values():
            self.all_locatables.update(targets)

        # A finite penalty to assign when no candidate nut/path is found
        self._penalty = max(10, len(self.goals) * 2)

    def _bfs_distances(self, start):
        """Return dict with shortest distances from `start` to reachable locatables."""
        # If start is unknown in the static graph, it's unreachable to others.
        if start not in self.all_locatables:
            # But still return {start: 0} so direct checks work.
            return {start: 0}
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.adjacency.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        """
        Compute the heuristic value for the given search node.

        Returns an integer estimate of remaining actions.
        """
        state = node.state

        # Quick check: if all goals already satisfied -> cost 0
        if self.goals <= state:
            return 0

        # Parse state facts: find for each nut its tightened locatable and loose-2 location.
        nut_tight_loc = {}   # nut -> locatable
        nut_loose2_loc = {}  # nut -> location (string)

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "tightened" and len(parts) == 3:
                nut, loc = parts[1], parts[2]
                nut_tight_loc[nut] = loc
                # record loc as locatable in arg set too
                self.all_locatables.add(loc)
            elif pred == "loose-2" and len(parts) == 3:
                nut, location = parts[1], parts[2]
                nut_loose2_loc[nut] = location

        # Precompute BFS distances for each distinct starting loc (nut tightened loc).
        bfs_cache = {}
        for loc in set(nut_tight_loc.values()):
            bfs_cache[loc] = self._bfs_distances(loc)

        total_cost = 0

        # For each goal (tightened-2 locatable location) compute an estimate.
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] != "tightened-2" or len(parts) != 3:
                # If there are non-tightened-2 goals, conservatively add penalty.
                # (The domain examples only use tightened-2 goals, so this is safe.)
                total_cost += self._penalty
                continue

            g_locatable, g_location = parts[1], parts[2]

            # If this exact goal is already in the current state it's 0 cost.
            if goal in state:
                continue

            # Candidate predecessor nodes b such that there's an adjacency edge b -> g_locatable.
            predecessors = self.predecessors.get(g_locatable, set())
            if not predecessors:
                # No static way to produce tightened-2 at g -> heavy penalty.
                total_cost += self._penalty
                continue

            best_for_goal = None

            # Consider every nut that could potentially produce tightened-2 for g.
            for nut, start_loc in nut_tight_loc.items():
                distances = bfs_cache.get(start_loc)
                if distances is None:
                    distances = self._bfs_distances(start_loc)
                    bfs_cache[start_loc] = distances

                # minimal distance from nut's tightened loc to any b in predecessors
                min_dist_to_pred = None
                for b in predecessors:
                    if b == start_loc:
                        d = 0
                    else:
                        d = distances.get(b)
                    if d is None:
                        continue
                    if min_dist_to_pred is None or d < min_dist_to_pred:
                        min_dist_to_pred = d

                if min_dist_to_pred is None:
                    # this nut cannot reach any required predecessor
                    continue

                # cost to make nut's loose-2 point to the goal location g_location:
                loose2_loc = nut_loose2_loc.get(nut)
                fix_loose2_cost = 0 if loose2_loc == g_location else 1

                # final step (walk/pickup_spanner) to produce tightened-2
                final_step = 1

                candidate = min_dist_to_pred + fix_loose2_cost + final_step

                if best_for_goal is None or candidate < best_for_goal:
                    best_for_goal = candidate

            if best_for_goal is None:
                # No nut could reach the predecessor for this goal -> penalty
                total_cost += self._penalty
            else:
                total_cost += best_for_goal

        # Ensure heuristic returns an integer
        return int(total_cost)