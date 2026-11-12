from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact string into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the "spanner" / "secret_domain" (spanner) domain.
    The heuristic estimates the number of actions required to achieve all goal
    facts of the form (tightened-2 <locatable> <location>). It reasons about
    how a tightened nut (predicate `tightened nut locatable`) can be moved
    across locatables (via edges defined by static relations loose/at/carrying/usable)
    to the target locatable and how a nut's loose-2 location (predicate `loose-2 nut location`)
    must be at the goal location in order to produce the final (tightened-2 ...) fact.
    The heuristic is deliberately optimistic about concurrency between different goals
    (it simply sums per-goal minimal estimates) and ignores some resource locking
    details (notably the exact sequencing of `link` toggles). This makes the heuristic
    fast to compute while still providing useful guidance.

    Assumptions
    -----------
    - Goals are only of the form (tightened-2 <locatable> <location>). Other goal predicates
      are ignored by this heuristic (but the planner still requires them).
    - The static relations loose, at, carrying, usable define possible single-step moves
      for a tightened nut from one locatable to a neighboring locatable. Concretely
      an operator that moves a tightened nut from v2 to v3 exists whenever one of:
        (loose v3 v2), (at v3 v2), (carrying v3 v2), (usable v3 v2)
      is true in the static facts. These create directed edges v2 -> v3 in a move graph.
    - Any nut that currently has a loose-2 location can be moved (with one action)
      between the available at-2 locations using the operator `tighten_nut`. Therefore,
      if a nut is not currently loose-2 at the goal location, we estimate cost +1
      to bring some nut's loose-2 to the goal location.
    - The heuristic ignores detailed `link` resource sequencing. If the target locatable
      does not currently hold `(link ...)` we add a small penalty (+1) to reflect that
      additional action(s) are usually needed to make the final tightening possible.

    Heuristic Initialization
    ------------------------
    The constructor extracts and indexes the static facts into efficient structures:
    - adjacency: directed adjacency dictionary mapping a locatable -> set(neighbor locatables)
      representing possible single-step moves for a tightened nut (edges v2 -> v3 when static
      has (loose v3 v2), (at v3 v2), (carrying v3 v2), or (usable v3 v2)).
    - at2_locations: set of location names that are `at-2` (these are legal locations for nuts).
    - Additionally, the goal list (task.goals) is stored to know which tightened-2 facts to evaluate.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node / state:
    1. Quickly check whether the state is already a goal state. If so, return 0.
    2. Parse the state into:
       - tightened_map: mapping nut -> locatable (from `(tightened nut locatable)` facts).
       - loose2_map: mapping nut -> location (from `(loose-2 nut location)` facts).
       - link_set: set of locatables currently having `(link ...)` (used for a small penalty).
       - tightened2_set: set of (locatable, location) pairs already satisfied in the state.
    3. For each goal (tightened-2 X L) that is not yet true:
       - We need a tightened nut to end up associated with locatable X while the nut's loose-2
         fact points to location L (so that a final operator creates `(tightened-2 X L)`).
       - Consider every nut N present in the state that has some tightened position Y.
         For that nut:
         a) Cost to place the nut's loose-2 at L:
            - 0 if loose2_map[N] == L (already at goal location).
            - 1 if loose2_map[N] exists but is different (we can use `tighten_nut` once to change it).
            - If a nut has no loose-2 fact, that nut is ignored for placement (in practice the
              problem instances provide loose-2 for nuts).
         b) Cost to move the tightened nut from its current locatable Y to target X:
            - Compute shortest path length from Y to X in the adjacency graph (each edge ~ one move-2 action).
            - If no path exists, treat this nut as infeasible for that goal.
         c) Final step cost:
            - +1 for the final operator that produces (tightened-2 X L).
            - +1 extra penalty if X is not in link_set (approximating the need to prepare `link X`).
         d) Total estimated cost for nut N = placement_cost + path_len + 1 + link_penalty.
       - Take the minimum total cost over all nuts as the estimated cost to achieve the particular goal (tightened-2 X L).
       - If no nut yields a finite cost, fall back to a conservative large cost for that goal.
    4. Sum the per-goal estimates to obtain the heuristic value for the state.
       (This ignores possible synergies where one action achieves multiple goals.)

    Notes on Admissibility and Performance
    --------------------------------------
    - This heuristic is not admissible in general (it may over- or under-estimate due to ignored
      resource interactions), but is designed to be informative and fast to compute.
    - All computations use BFS on a small adjacency graph built from static facts; complexity
      is low and the heuristic should be efficient to evaluate during greedy best-first search.
    """

    def __init__(self, task):
        # store goals for later
        self.goals = task.goals
        static_facts = task.static

        # Build adjacency graph for locatables. Edges v2 -> v3 exist if static contains
        # (loose v3 v2), (at v3 v2), (carrying v3 v2), or (usable v3 v2).
        self.adjacency = {}  # node -> set(neighbors)
        self.at2_locations = set()
        self.all_locatables = set()

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ("loose", "at", "carrying", "usable"):
                # parts are (pred a b) and create directed edge b -> a
                if len(parts) >= 3:
                    a, b = parts[1], parts[2]
                    self.all_locatables.add(a)
                    self.all_locatables.add(b)
                    self.adjacency.setdefault(b, set()).add(a)
                    # ensure nodes exist
                    self.adjacency.setdefault(a, set())
            elif pred == "at-2":
                if len(parts) >= 2:
                    self.at2_locations.add(parts[1])

        # Make sure any referenced locatable has an adjacency entry
        for node in list(self.all_locatables):
            self.adjacency.setdefault(node, set())

        # Pre-process goal list into pairs for faster use
        self.goal_pairs = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "tightened-2" and len(parts) >= 3:
                locatable = parts[1]
                location = parts[2]
                self.goal_pairs.append((locatable, location))

        # A conservative large cost for unreachable situations
        self.INF_COST = 1000

    def _shortest_path_length(self, start, target):
        """Breadth-first search on the adjacency graph. Returns shortest path length (edges)
        from start to target, or INF_COST if not reachable."""
        if start == target:
            return 0
        if start not in self.adjacency or target not in self.adjacency:
            return self.INF_COST
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nbr in self.adjacency.get(node, ()):
                if nbr == target:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        return self.INF_COST

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given node/state.
        Returns an integer estimate of the number of actions needed to reach the goals.
        """
        state = node.state

        # Quick goal check: if all goals are already satisfied, return 0.
        if self.goals <= state:
            return 0

        # Parse dynamic facts from the current state.
        tightened_map = {}   # nut -> locatable (from (tightened nut locatable))
        loose2_map = {}      # nut -> location (from (loose-2 nut location))
        link_set = set()     # locatables that currently have (link ...)
        tightened2_set = set()  # set of (locatable, location) pairs already satisfied

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "tightened" and len(parts) >= 3:
                nut, loc = parts[1], parts[2]
                tightened_map[nut] = loc
            elif pred == "loose-2" and len(parts) >= 3:
                nut, location = parts[1], parts[2]
                loose2_map[nut] = location
            elif pred == "link" and len(parts) >= 2:
                link_set.add(parts[1])
            elif pred == "tightened-2" and len(parts) >= 3:
                locatable, location = parts[1], parts[2]
                tightened2_set.add((locatable, location))

        # If there are no nuts with tightened information, return a fallback estimate.
        if not tightened_map:
            # No tightened nuts => cannot produce tightened-2 facts; return high but finite cost.
            return len(self.goal_pairs) * 5

        total_cost = 0

        # Precompute whether there exists at least one nut that has some loose-2 location
        any_loose2_exists = bool(loose2_map)

        for (target_locatable, target_location) in self.goal_pairs:
            # If already satisfied, cost 0 for this goal.
            if (target_locatable, target_location) in tightened2_set:
                continue

            best_goal_cost = self.INF_COST

            # Try every nut as a candidate to produce the tightened-2 for (target_locatable, target_location)
            for nut, tightened_pos in tightened_map.items():
                # Cost to have this nut's loose-2 at the target location:
                placement_cost = self.INF_COST
                if nut in loose2_map:
                    if loose2_map[nut] == target_location:
                        placement_cost = 0
                    else:
                        # We can use tighten_nut once to move the loose-2 marker to the desired location.
                        placement_cost = 1
                else:
                    # This nut has no loose-2 recorded; ignore it (or consider infeasible).
                    placement_cost = self.INF_COST

                if placement_cost >= self.INF_COST:
                    continue  # nut cannot be used for placement

                # Cost to move the tightened nut from its current locatable to the target locatable.
                path_len = self._shortest_path_length(tightened_pos, target_locatable)
                if path_len >= self.INF_COST:
                    continue  # cannot reach this target locatable with this nut

                # Final action to create (tightened-2 ...) is counted as 1.
                final_action_cost = 1

                # If the target locatable currently lacks a link, add a small penalty to reflect
                # the need to manipulate link resource before the final action.
                link_penalty = 0 if target_locatable in link_set else 1

                total_for_nut = placement_cost + path_len + final_action_cost + link_penalty

                if total_for_nut < best_goal_cost:
                    best_goal_cost = total_for_nut

            # If we could not find any nut with a finite cost (rare in valid instances),
            # but some loose-2 exists somewhere, we still allow bringing a nut and moving it:
            if best_goal_cost >= self.INF_COST and any_loose2_exists:
                # Conservative fallback: one placement + assume path of length proportional to nodes
                # (but bounded), plus final and link penalty.
                # Use 1 (placement) + estimated path (max 3) + 1 final + link penalty
                link_penalty = 0 if target_locatable in link_set else 1
                best_goal_cost = 1 + 3 + 1 + link_penalty

            # If still INF, give a finite but large cost
            if best_goal_cost >= self.INF_COST:
                best_goal_cost = 10  # conservative fallback

            total_cost += best_goal_cost

        # Ensure heuristic is integer and finite
        if total_cost >= self.INF_COST:
            total_cost = sum(10 for _ in self.goal_pairs)

        return int(total_cost)