from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the parts of a PDDL fact string by removing surrounding parentheses and splitting.
    Example: "(z6o4oitz a b)" -> ["z6o4oitz", "a", "b"].
    """
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (mtfkspuf) domain.
    The main observable structure in this domain is a connectivity relation
    between locations of type hmhptaxe encoded by the predicate z6o4oitz,
    and facts jft9ynda(<thing>, <location>) that indicate where an entity
    currently is. The action k12p0czo moves a jft9ynda fact along a
    z6o4oitz edge (i.e., it implements a single-step location transition).
    Other operators (sw2fg9ql and vyn92k55) can create or copy jft9ynda facts
    under certain ownership/relationship preconditions (uypgj7u3, yarbdfgc,
    jf4qno7u). The heuristic leverages the location graph to estimate how
    many single-step moves (k12p0czo) are necessary and is optimistic about
    one-step copy/creation opportunities when a direct copy is plausibly available.

    Assumptions
    -----------
    - The relevant goals for this domain are typically of the form
      (jft9ynda <obj> <loc>), i.e., placing an object at a location.
    - The k12p0czo operator moves a jft9ynda fact along a directed
      z6o4oitz edge, and therefore the shortest-path distance in the
      directed z6o4oitz graph approximates the number of k12p0czo actions
      required to move an object between locations.
    - If an object currently has no jft9ynda fact, a jft9ynda for it may
      be created/copied from a related hodp29q0 object when a uypgj7u3
      relation exists that links the g83opnlj (object in goals) to some
      hodp29q0 that already has a jft9ynda. We treat such copying (via
      sw2fg9ql) optimistically as costing one extra action on top of the
      movement cost to bring that source hodp29q0 to the goal location.
    - We ignore some subtle preconditions required by sw2fg9ql/vyn92k55
      (additional jf4qno7u / yarbdfgc requirements) and treat the copy
      possibility optimistically (i.e., if the structural requirements
      appear satisfied in the state we allow a +1 cost to copy).

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - self.goals: the set of goal fact strings.
    - The directed adjacency graph of hmhptaxe locations from static facts
      of the form (z6o4oitz A B). Edges are stored as adjacency[A] contains B.
    - The set of uypgj7u3 relations present in the initial static info and
      also available dynamic facts (we consider both static and dynamic
      presence at evaluation time).
    The adjacency graph is used for shortest-path computations between locations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given planner node (state):
    1. If all goals are satisfied in state, return 0.
    2. Build a map of current jft9ynda facts (subject -> location) from the state.
       These represent entities whose location can be advanced by k12p0czo.
    3. Also collect the uypgj7u3 relations from the current state (they link
       some g83opnlj entities to hodp29q0 entities and can make copying possible).
    4. For each goal fact:
       - If the goal fact is already true in state, it contributes 0.
       - If the goal is a jft9ynda(<obj>, <goal_loc>) not true in state,
         compute candidate costs:
           (a) If <obj> currently has a jft9ynda location, compute the directed
               shortest-path distance (number of z6o4oitz edges) from that
               current location to <goal_loc>. Each step corresponds to one
               k12p0czo action.
           (b) If <obj> can be copied/created from a related hodp29q0 entity
               (i.e., there is a uypgj7u3 (<obj> <source>) present and
               <source> has a jft9ynda at some location L), then compute
               the distance from L to <goal_loc> and add +1 to account for
               the optimistic copy action (sw2fg9ql). This models the
               sequence "move the source to goal, then copy" (or if the
               source is already at goal, just copy => cost 1).
           (c) If neither (a) nor (b) yields a reachable path, we fall back to
               a conservative finite estimate based on the number of known
               locations (this guarantees finiteness).
       - Use the minimum candidate cost for this goal and add it to the
         heuristic total (we sum individual goal estimates).
    5. Return the summed estimate. The heuristic is intentionally optimistic
       (it may under-estimate because it ignores some preconditions), but it
       is cheap to compute and guides the search toward satisfying
       jft9ynda-location goals by following the location graph.

    Implementation notes
    --------------------
    - The heuristic only requires collections.deque and the planner's Heuristic base class.
    - get_parts(...) is used to parse PDDL facts; this ensures surrounding
      parentheses are ignored (checklist item 1).
    - The heuristic returns 0 exactly for goal states (checklist item 2).
    - A finite fallback estimate is provided for unreachable cases
      (checklist item 3).
    - Static z6o4oitz facts are extracted during initialization
      (checklist item 5).
    """

    def __init__(self, task):
        # store goals and static info
        self.goals = task.goals
        self.static = task.static

        # Build adjacency graph for hmhptaxe locations from static z6o4oitz facts.
        # The graph is directed: an edge (A -> B) exists if (z6o4oitz A B) is present.
        self.adj = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "z6o4oitz" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)

        # The set of all known locations (nodes) in the adjacency graph.
        self.all_locations = set(self.adj.keys())
        for dests in self.adj.values():
            self.all_locations.update(dests)

    def _bfs_distance(self, start, target):
        """Compute directed shortest-path distance (number of edges) from start to target.
        Returns an integer distance if reachable, else None.
        """
        if start == target:
            return 0
        if start not in self.all_locations:
            return None
        visited = set()
        q = deque([(start, 0)])
        visited.add(start)
        while q:
            node, dist = q.popleft()
            for nei in self.adj.get(node, ()):
                if nei == target:
                    return dist + 1
                if nei not in visited:
                    visited.add(nei)
                    q.append((nei, dist + 1))
        return None

    def __call__(self, node):
        """Compute heuristic estimate for the planner node."""
        state = node.state

        # If goals already reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Map current jft9ynda facts: subject -> location
        jft_map = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "jft9ynda" and len(parts) >= 3:
                subj, loc = parts[1], parts[2]
                jft_map[subj] = loc

        # Collect uypgj7u3 relations present in the current state (links g83opnlj -> hodp29q0)
        uyp_links = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "uypgj7u3" and len(parts) >= 3:
                # link: (uypgj7u3 g83opnlj hodp29q0)
                uyp_links.add((parts[1], parts[2]))

        # Also consider static uypgj7u3 relations (some domains may encode them statically)
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "uypgj7u3" and len(parts) >= 3:
                uyp_links.add((parts[1], parts[2]))

        total_cost = 0
        # Fallback large finite cost in absence of connectivity
        fallback_cost = max(1, len(self.all_locations)) + 2

        # Evaluate each goal separately and sum optimistic estimates.
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            pred = parts[0]
            if pred != "jft9ynda" or len(parts) < 3:
                # For non-jft9ynda goals, use a generic cost of 1 if not true.
                if goal not in state:
                    total_cost += 1
                continue

            subj, goal_loc = parts[1], parts[2]
            if goal in state:
                continue  # already satisfied

            candidate_costs = []

            # Case A: the subject already has a jft9ynda location -> move along z6o4oitz edges
            if subj in jft_map:
                cur_loc = jft_map[subj]
                dist = self._bfs_distance(cur_loc, goal_loc)
                if dist is not None:
                    candidate_costs.append(dist)
                else:
                    # no directed path: large but finite estimate
                    candidate_costs.append(fallback_cost)

            # Case B: subject could be created/copied from a linked hodp29q0 (uypgj7u3 relation)
            # For any (subj, source) in uyp_links, if source has a jft9ynda location,
            # then we can (optimistically) move the source and then copy (+1).
            for (g_obj, source) in uyp_links:
                if g_obj != subj:
                    continue
                if source in jft_map:
                    src_loc = jft_map[source]
                    dist_src = self._bfs_distance(src_loc, goal_loc)
                    if dist_src is not None:
                        # +1 to account for sw2fg9ql (optimistic copy)
                        candidate_costs.append(dist_src + 1)
                    else:
                        candidate_costs.append(fallback_cost + 1)

            # If we found no optimistic candidate (subject not present and no links),
            # try a generic conservative estimate: distance from any known jft9ynda location
            # to the goal plus one to create/copy.
            if not candidate_costs:
                # try distances from any jft9ynda presence
                min_dist = None
                for loc in jft_map.values():
                    d = self._bfs_distance(loc, goal_loc)
                    if d is not None and (min_dist is None or d < min_dist):
                        min_dist = d
                if min_dist is not None:
                    # we assume at least one extra action to create/copy the needed jft9ynda
                    candidate_costs.append(min_dist + 1)
                else:
                    candidate_costs.append(fallback_cost)

            # Use the minimum optimistic candidate for this particular goal
            total_cost += min(candidate_costs)

        # The heuristic is an optimistic sum of per-goal estimates and therefore finite.
        return total_cost