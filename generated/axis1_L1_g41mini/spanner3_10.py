from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components from a PDDL fact (remove parentheses and split)."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards ('*').

    Example:
        match("(u4srx1nb bob shed)", "u4srx1nb", "*", "shed") returns True.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    This heuristic estimates the minimal number of actions required to reach the goal
    state by focusing on the essential subgoals related to 'g59iwyxf' facts for nuts,
    and the necessary corresponding preconditions including moving tools/items 
    (such as spanners) through locations, and enabling the goal effects by triggering 
    the final action that changes predicates 'bu1om0m1' and 'g0brvc06' into the goal state.

    Assumptions
    -----------
    - Each nut (?) of type rupjajag must have the predicate (g59iwyxf nut) true in the goal.
    - To achieve (g59iwyxf nut), the "kjsqfni8" action must be applied.
    - Preconditions for "kjsqfni8": 
        - (u4srx1nb ?v7he5 ?r2bzp) for some ?v7he5 of ilfxqhcc and some ?r2bzp of i6azntdy
        - (u4srx1nb ?gdh2s ?r2bzp) for the nut ?gdh2s and same location ?r2bzp
        - (yud7t3vb ?v7he5 ?aqz3v) linking ilfxqhcc to e2zu8y93 (spanner)
        - (g0brvc06 ?aqz3v) for spanner
        - (bu1om0m1 ?gdh2s) for nut
    - Opening/transporting spanners involves "n9sbfmrs" actions that move the tool's position.
    - (yud7t3vb ?v7he5 ?aqz3v) can be established via "moi6wnfu" action,
      which requires both the ilfxqhcc and the spanner to be at the same location.
    - "kxokb5hx" predicates encode connectivity between locations (likely a graph).

    Heuristic Initialization
    ------------------------
    - Extract goal nuts from goal facts with (g59iwyxf nut).
    - Extract static connectivity graph from facts with (kxokb5hx loc1 loc2).
    - Extract all objects by type (locations, spanners, nuts, ilfxqhcc).
      Though full typing info might not be needed, it helps clarify roles.
    - Build adjacency info for location connectivity for shortest path computations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify which nuts still need to satisfy (g59iwyxf nut) (i.e. goal nuts not yet in state).
    2. For each nut to be "goal-enabled", ensure that predicate (bu1om0m1 nut) holds at some location.
    3. To apply "kjsqfni8" to achieve (g59iwyxf nut):
       a. The nut and some ilfxqhcc (e.g., 'bob') and some spanner must be co-located at a location ?r2bzp.
       b. The ilfxqhcc must have (yud7t3vb ilfxqhcc spanner).
       c. The spanner must currently have (g0brvc06 spanner) true.
       d. The nut must have (bu1om0m1 nut).
    4. The ilfxqhcc can "carry" the spanner (yud7t3vb ilfxqhcc spanner) by "moi6wnfu" action,
       which requires both at same location.
    5. Spanners can be moved between adjacent locations with "n9sbfmrs" actions by "u4srx1nb ilfxqhcc location".
    6. So, the heuristic will:
       - For each nut still needing the (g59iwyxf nut) goal:
         * Count 1 action for the final "kjsqfni8".
         * Estimate actions needed to move a spanner from its current location (holding (g0brvc06)) 
           to the location of the nut.
         * Possibly estimate moving the ilfxqhcc ("bob") to the nut location if needed.
         * Ensure that (yud7t3vb bob spanner) is true by counting at least 1 action "moi6wnfu".
       - Sum those costs over all nuts.

    7. This computation requires:
       - Distances computed over the connectivity graph for traveling between locations (shortest path).
       - Actual current locations for bob, spanners, and nuts.
       - The number of moves to transport spanners and have bob pick it up, position himself and spanner
         well enough to trigger "kjsqfni8" per nut.

    8. This heuristic is not necessarily admissible but aims to keep cost reasonable and consistent
       with problem structure, capturing movement and setup costs.

    Implementation notes:
    - For efficiency, precompute location adjacency from static facts.
    - Use simple BFS shortest path in the connectivity graph to compute distances.
    - Assume 'bob' is a single ilfxqhcc character (in examples only 'bob' present).
    - If multiple nuts, sum independently as heuristic over-approximation.
    - Use presence/absence of relevant predicates to estimate needed actions.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static_facts = task.static

        # Extract connectivity (kxokb5hx loc1 loc2) to build graph (undirected)
        self.adj = {}
        for fact in self.static_facts:
            if match(fact, "kxokb5hx", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.adj.setdefault(loc1, set()).add(loc2)
                self.adj.setdefault(loc2, set()).add(loc1)

        # Extract goal nuts (those with (g59iwyxf nut))
        self.goal_nuts = set()
        for fact in self.goals:
            if match(fact, "g59iwyxf", "*"):
                self.goal_nuts.add(get_parts(fact)[1])

        # We will identify:
        # - Bob(s) (all ilfxqhcc that appear in the state with u4srx1nb)
        # - Spanners (e2zu8y93 objects, from predicate occurrences)
        # - Nuts (rupjajag objects subset is in goals)
        # This will be inferred dynamically for current state as needed.

    def shortest_path_length(self, start, goal):
        """Compute shortest number of hops between start and goal in adjacency graph."""
        if start == goal:
            return 0
        visited = {start}
        frontier = [start]
        dist = 0
        while frontier:
            dist += 1
            next_frontier = []
            for node in frontier:
                for neigh in self.adj.get(node, []):
                    if neigh == goal:
                        return dist
                    if neigh not in visited:
                        visited.add(neigh)
                        next_frontier.append(neigh)
            frontier = next_frontier
        # No path found means large cost; return large number
        return 1000

    def find_u4srx1nb_entities(self, state):
        """Return dict: object -> location for facts (u4srx1nb obj location)."""
        results = {}
        for fact in state:
            if match(fact, "u4srx1nb", "*", "*"):
                pred, obj, loc = get_parts(fact)
                results[obj] = loc
        return results

    def find_yud7t3vb(self, state):
        """Return dict: ilfxqhcc -> spanner for (yud7t3vb ilfxqhcc spanner) facts."""
        results = {}
        for fact in state:
            if match(fact, "yud7t3vb", "*", "*"):
                _, ilf, spanner = get_parts(fact)
                results[ilf] = spanner
        return results

    def has_fact(self, state, pred, *args):
        """Returns True if predicate with args exists in state."""
        pattern = "({} {})".format(pred, " ".join(args))
        # Direct exact match or fnmatch on predicate with wildcards:
        for fact in state:
            if fnmatch(fact, pattern):
                return True
        return False

    def __call__(self, node):
        """
        Compute heuristic estimate.

        Return 0 if goals are reached.

        Otherwise, estimate how many actions needed minimally to enable (g59iwyxf nut)
        for all nuts in the goal not yet satisfied.

        Estimate includes movement of spanners to nuts, bob/spanner co-location,
        setting (yud7t3vb bob spanner), and applying final "kjsqfni8" action.
        """

        state = node.state

        # If goal reached, heuristic is zero
        if self.goals <= state:
            return 0

        # Extract locations of all objects with (u4srx1nb obj loc)
        obj_locations = self.find_u4srx1nb_entities(state)

        # Extract (yud7t3vb bob spanner) mappings
        carried_spanners = self.find_yud7t3vb(state)

        # Identify bob(s): ilfxqhcc objects -- appears in u4srx1nb facts or yud7t3vb facts filtered
        ilfxqhcc_in_state = set()
        for fact in state:
            if match(fact, "u4srx1nb", "*", "*"):
                obj = get_parts(fact)[1]
                # The domain types suggest ilfxqhcc are a subset; no direct typing here, assume bob from examples.
                # We just get all potential ilfxqhcc by checking who occurs in yud7t3vb
                pass
        ilfxqhcc_in_state.update(carried_spanners.keys())

        # Practical heuristic: pick one ilfxqhcc (e.g., bob), or fallback to first.
        if ilfxqhcc_in_state:
            ilf_person = next(iter(ilfxqhcc_in_state))
        else:
            # Fallback if none found, might happen in partial states
            ilf_person = None

        # Identify all spanners: those appearing in (g0brvc06 spanner) facts or as second arg in yud7t3vb
        spanners = set()
        for fact in state:
            if match(fact, "g0brvc06", "*"):
                sp = get_parts(fact)[1]
                spanners.add(sp)
            elif match(fact, "yud7t3vb", "*", "*"):
                sp = get_parts(fact)[2]
                spanners.add(sp)

        # Identify nuts: those in goal with (g59iwyxf nut), and appearing in (bu1om0m1 nut)
        nuts_needed = []
        for nut in self.goal_nuts:
            # Check if satisfied:
            if ("(g59iwyxf {})".format(nut)) not in state:
                nuts_needed.append(nut)

        # Locate nuts and check if (bu1om0m1 nut) true
        nuts_bu1om0m1 = set()
        for fact in state:
            if match(fact, "bu1om0m1", "*"):
                nuts_bu1om0m1.add(get_parts(fact)[1])

        # Heuristic sum over nuts not yet goal achieved
        total_cost = 0

        # For each nut requiring goal predicate:
        for nut in nuts_needed:
            # 1. The final action "kjsqfni8" needs 1 action
            cost = 1

            # 2. Nut location
            nut_loc = obj_locations.get(nut, None)
            if nut_loc is None:
                # No known location -> very high cost
                cost += 100
                total_cost += cost
                continue

            # 3. Check if (bu1om0m1 nut) holds; if not, probably needs action (not in model),
            # we assume 1 extra action to fix this
            if nut not in nuts_bu1om0m1:
                cost += 1  # penalty for needing to fix this

            # 4. Find a spanner that has (g0brvc06 spanner).
            spanner_with_g0brvc06 = None
            for sp in spanners:
                if ("(g0brvc06 {})".format(sp)) in state:
                    spanner_with_g0brvc06 = sp
                    break

            if spanner_with_g0brvc06 is None:
                # No spanner ready; large cost
                cost += 100
                total_cost += cost
                continue

            spanner_loc = obj_locations.get(spanner_with_g0brvc06, None)
            if spanner_loc is None:
                cost += 100
                total_cost += cost
                continue

            # 5. Location of ilfxqhcc (bob or relevant)
            ilf_loc = obj_locations.get(ilf_person, None)
            if ilf_loc is None:
                # Account for cost to bring bob to somewhere by ignoring location difference
                ilf_loc = nut_loc  # pessimistically assume already there

            # 6. Compute shortest paths for transporting the spanner to the nut location
            # using "n9sbfmrs" which moves spanner along connected locations.

            spanner_to_nut_dist = self.shortest_path_length(spanner_loc, nut_loc)
            # Moving spanner one location costs 1 action each.

            # 7. To have (yud7t3vb ilfxqhcc spanner), "moi6wnfu" action must be applied at same location:
            # ilfxqhcc and spanner must be at the same place. We estimate:
            # cost to bring ilfxqhcc to spanner's location or spanner to ilfxqhcc, whichever closer,
            # then 1 action for "moi6wnfu".
            ilf_to_spanner_dist = self.shortest_path_length(ilf_loc, spanner_loc)
            nut_to_ilf_dist = self.shortest_path_length(nut_loc, ilf_loc)

            # Minimum moves is to bring both together at one location before final.
            # We approximate by:
            #  move_spanner + move_ilf_to_spanner + 1 moi6wnfu
            # Alternatively, spanner moved directly to nut loc and ilf moved there.

            # More precise approximation:
            # Move spanner to nut
            cost += spanner_to_nut_dist   # moves of spanner
            # Move ilf to nut
            cost += nut_to_ilf_dist       # moves of ilfxqhcc
            # One moi6wnfu to establish (yud7t3vb ilfxqhcc spanner)
            cost += 1

            # 8. Now at nut location ilf and spanner are co-located and spanner is (g0brvc06)
            # so final kjsqfni8 can be applied (already counted).

            total_cost += cost

        return total_cost
