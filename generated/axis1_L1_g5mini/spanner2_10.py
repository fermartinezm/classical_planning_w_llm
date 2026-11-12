from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the components of a PDDL fact string by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (e2zu8y93). The domain models
    a worker (person) that can pick up spanners and use them to "fix" nuts.
    Goals are facts of the form (g59iwyxf <nut>). To produce such a goal for a nut
    the action `kjsqfni8` must be applied. That action requires:
      - a person and the nut at the same location,
      - the person to "have" a spanner (yud7t3vb person spanner),
      - the spanner to be usable (g0brvc06 spanner),
      - the nut to be present (bu1om0m1 nut).
    A spanner is acquired by the `moi6wnfu` action which requires the person and
    spanner to be at the same location (and consumes the spanner-location fact
    and produces yud7t3vb). The person moves between adjacent locations using
    `n9sbfmrs` (one action per adjacency edge).
    The heuristic estimates the number of actions needed to satisfy all remaining
    nut goals by greedily simulating a sequence of: move-to-spanner (if needed),
    pick-spanner, move-to-nut, and use-spanner (kjsqfni8). It accounts for
    spanners already held by the person and for the directed adjacency graph
    defined by static (kxokb5hx ...) facts.

    Assumptions
    -----------
    - There is at least one person (ilfxqhcc) in the instance. The heuristic
      identifies person(s) as objects appearing in location facts (u4srx1nb)
      that are not classified as spanners or nuts.
    - Spanners are identified by the predicate (g0brvc06 <spanner>) and/or
      by appearing as the second argument in (yud7t3vb <person> <spanner>).
    - Nuts are identified by (bu1om0m1 <nut>) and goals of the form
      (g59iwyxf <nut>).
    - Movement between locations follows the directed edges in static facts
      (kxokb5hx <from> <to>); one invocation of n9sbfmrs moves the person
      one edge and costs 1.
    - Using kjsqfni8 consumes the "goodness" of the spanner (g0brvc06) so a
      spanner can be used only once for this purpose.
    - If no usable spanner is available for some remaining nut, the heuristic
      returns a large finite value (indicating a likely unsolvable or expensive state).

    Heuristic Initialization
    ------------------------
    - Extract the set of goal nuts from the task goals (predicate g59iwyxf).
    - Extract the adjacency graph between location nodes from static facts
      (kxokb5hx).
    - Keep the static facts for distance computations.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state:
    1. If all goals are already satisfied in the state, return 0.
    2. Parse the state to obtain:
       - object -> location mapping from u4srx1nb facts,
       - which spanners are currently "good" (g0brvc06),
       - which nuts are present (bu1om0m1),
       - which spanners are already held by the person (yud7t3vb).
    3. Identify the person (one of the objects present in location facts that is
       neither a nut nor a spanner). If multiple persons exist, pick one (the
       heuristic simulates actions for a single worker).
    4. Build a directed adjacency graph from the static kxokb5hx facts and use
       BFS to compute shortest-path distances between locations on demand,
       caching computed distances for efficiency.
    5. For every goal nut not yet achieved, compute the minimal estimated
       number of actions required to "fix" it given the current person location
       and the currently available good spanners. The basic cost model for
       fixing a nut using a spanner s is:
         - if s is already held by the person and is still good:
             cost = distance(person_loc, nut_loc) + 1  # travel + kjsqfni8
         - else (spanner s at location sp_loc):
             cost = distance(person_loc, sp_loc) + 1   # travel + pick (moi6wnfu)
                  + distance(sp_loc, nut_loc) + 1     # travel + use (kjsqfni8)
       Distances that are unreachable are treated as infinite and such pairings
       are ignored.
    6. Greedily pick the nut-spanner pair with minimal cost, add that cost to
       the running total, update the simulated person location to the nut's
       location, and mark the selected spanner as consumed (cannot be reused).
    7. Repeat step 6 until all remaining nuts are processed or no feasible
       pairing exists.
    8. Return the accumulated cost. If any remaining nut has no feasible pairing,
       return a large finite value.

    The heuristic is intentionally not admissible (it may underestimate or
    overestimate) but aims to provide a compact, informed estimate that is
    inexpensive to compute and helpful in guiding greedy best-first search.
    """

    def __init__(self, task):
        """Extract static adjacency info and goal nuts from the task."""
        self.goals = task.goals
        self.static = task.static

        # adjacency: directed graph mapping location -> list of neighbor locations
        self.adjacency = defaultdict(list)
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "kxokb5hx" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.adjacency[src].append(dst)

        # goal nuts: extract object names from goal facts of predicate g59iwyxf
        self.goal_nuts = set()
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "g59iwyxf" and len(parts) >= 2:
                self.goal_nuts.add(parts[1])

        # cache for distances between locations (BFS results)
        self._distance_cache = {}

    def _bfs_distances_from(self, src):
        """Return BFS distances from src to all reachable locations (cache results)."""
        if src in self._distance_cache:
            return self._distance_cache[src]
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.adjacency.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._distance_cache[src] = dist
        return dist

    def _distance(self, a, b):
        """Directed shortest-path distance from location a to b. Return None if unreachable."""
        if a == b:
            return 0
        dmap = self._bfs_distances_from(a)
        return dmap.get(b, None)

    def __call__(self, node):
        """Compute the heuristic estimate for the given search node."""
        state = node.state

        # 0 for actual goal states
        if self.goals <= state:
            return 0

        # Parse state facts
        loc_of = {}        # object -> location (from u4srx1nb facts)
        good_spanners = set()  # spanner objects that currently have g0brvc06
        nuts_present = set()   # nuts that currently have bu1om0m1
        held_spanners_by_person = defaultdict(set)  # person -> set(spanners) from yud7t3vb

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "u4srx1nb" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                loc_of[obj] = loc
            elif pred == "g0brvc06" and len(parts) >= 2:
                sp = parts[1]
                good_spanners.add(sp)
            elif pred == "bu1om0m1" and len(parts) >= 2:
                nut = parts[1]
                nuts_present.add(nut)
            elif pred == "yud7t3vb" and len(parts) >= 3:
                person, sp = parts[1], parts[2]
                held_spanners_by_person[person].add(sp)

        # Determine person(s): objects present in loc_of but not spanners and not nuts
        spanners_all = set(good_spanners) | set(s for sset in held_spanners_by_person.values() for s in sset)
        # There might be spanners that are neither in good_spanners nor in yud7t3vb (rare),
        # but we treat spanners primarily by presence in g0brvc06 or yud7t3vb.
        persons = [o for o in loc_of.keys() if o not in spanners_all and o not in nuts_present]

        # If no person can be identified, return a moderate large finite cost (state likely invalid)
        if not persons:
            return 100000

        # Choose one person to simulate (heuristic chooses first)
        person = sorted(persons)[0]
        person_loc = loc_of.get(person, None)
        if person_loc is None:
            return 100000

        # Determine remaining goal nuts (those goals not yet achieved in the current state)
        remaining_nuts = [n for n in self.goal_nuts if f"(g59iwyxf {n})" not in state]

        # Build current available good spanners (only those with g0brvc06)
        available_spanners = set(good_spanners)

        # Also consider spanners that are held by the person but must be good to be usable
        # (they should appear in good_spanners to be usable)
        held_spanners = held_spanners_by_person.get(person, set()) & available_spanners

        total_cost = 0

        # Greedily assign spanners to nuts minimizing incremental cost
        # If unreachable, return a large finite penalty
        LARGE = 100000

        # For efficiency cache distances between locations already computed
        # We'll simulate consumption of spanners: each used spanner is removed from available_spanners
        for _ in range(len(remaining_nuts)):
            if not remaining_nuts:
                break

            best_option = None  # (cost, nut, spanner, held_flag, new_person_loc)
            # iterate all remaining nuts and available spanners to find best pairing
            for nut in remaining_nuts:
                nut_loc = loc_of.get(nut, None)
                if nut_loc is None:
                    # if nut location unknown, skip (unreachable)
                    continue

                # If the person already holds some usable spanners, consider those first
                for sp in list(available_spanners):
                    sp_held = sp in held_spanners
                    if sp_held:
                        # cost = distance(person_loc -> nut_loc) + 1 (kjsqfni8)
                        d = self._distance(person_loc, nut_loc)
                        if d is None:
                            continue
                        cost = d + 1
                    else:
                        # sp not held: sp must have a location to pick it up
                        sp_loc = loc_of.get(sp, None)
                        if sp_loc is None:
                            continue
                        d1 = self._distance(person_loc, sp_loc)
                        d2 = self._distance(sp_loc, nut_loc)
                        if d1 is None or d2 is None:
                            continue
                        # travel to spanner (d1), pick (1), travel to nut (d2), use (1)
                        cost = d1 + 1 + d2 + 1

                    if best_option is None or cost < best_option[0]:
                        best_option = (cost, nut, sp, sp_held, nut_loc)

            if best_option is None:
                # No feasible pairing found for remaining nuts -> large penalty
                total_cost += LARGE
                break

            cost, chosen_nut, chosen_sp, was_held, new_person_loc = best_option
            total_cost += cost

            # update simulated state: person now at nut location after kjsqfni8
            person_loc = new_person_loc

            # consume chosen spanner: remove from available_spanners and from held_spanners
            if chosen_sp in available_spanners:
                available_spanners.remove(chosen_sp)
            if chosen_sp in held_spanners:
                held_spanners.remove(chosen_sp)

            # remove the nut from remaining list
            remaining_nuts.remove(chosen_nut)

        return total_cost