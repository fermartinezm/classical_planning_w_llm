from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic estimates the minimum number of actions needed to achieve the goal condition(s) in the secret_domain.
    The typical goal predicate is (g59iwyxf ?nut) signifying that each nut (rupjajag) must have the predicate g59iwyxf true.
    The heuristic considers the current state of nuts, spanners, locations, and relevant intermediate predicates to estimate the number of
    moves and manipulations (swaps of locations, usage of spanners, and unlocking steps) needed to reach the goal.

    Assumptions
    -----------
    - The goal requires all nuts to satisfy (g59iwyxf ?nut).
    - Nuts start at some locations and need to be "activated" via a chain of actions.
    - To activate a nut, it must be:
      1) Held at a location (u4srx1nb ?nut ?loc),
      2) The spanner at that location must be "ready" (g0brvc06 spanner),
      3) The spanner must have the correct attributes (bu1om0m1 ?nut, yud7t3vb ?wrench ?spanner),
      4) There is a connectivity graph on locations via (kxokb5hx ?loc1 ?loc2),
         allowing movement of wrenches (?v7he5) between locations.
    - The primary operations are wrench moves between locations, activating spanners onto e2zu8y93 objects, and finally applying a spanner to unlock a nut.
    - Every wrench move requires one action.
    - Every activation or unlocking requires one action.
    - Wrenches can be moved only between adjacent locations (kxokb5hx).
    - The heuristic only counts estimated minimum wrench movements plus the minimal number of required activation/unlocking actions.

    Heuristic Initialization
    ------------------------
    - Parse static facts to build an undirected graph representing location adjacency from (kxokb5hx ?loc1 ?loc2).
    - Extract the sets of nuts (rupjajag), spanners (e2zu8y93), wrenches/people (ilfxqhcc), and locations (i6azntdy) from objects in static/facts.
      (Note: For the heuristic, object extraction uses the static facts and current state facts if needed; adjacency graph is essential.)
    - Identify goal nuts from task goals.
    - Store wrench and their current locations from the initial state.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each nut required to satisfy (g59iwyxf nut) in goal but not yet satisfied in current state:
        1. Identify the current location of the nut.
        2. Identify a spanner that is ready (g0brvc06 spanner) and co-located with some wrench.
        3. Estimate the minimal wrench moves needed to move a wrench from its current location to the nut's location:
           - Use BFS on location graph to find shortest path distance between wrench location and nut location.
        4. Account for the minimal number of actions to:
           - Move wrench(s) to adjacent locations to the nut.
           - Possibly activate wrench onto the spanner if not yet activated (using moi6wnfu).
           - Use the spanner to unlock the nut (using kjsqfni8).
        5. Sum these estimates for each nut (heuristic is additive over nuts).
    
    Details:
    - Assume each wrench movement step costs 1 action.
    - Each wrench-to-spanner activation costs 1 action.
    - Each unlocking action costs 1 action.
    - If multiple nuts share wrenches/spanners, no concurrency accounted (overestimates, admissibility not required).
    - If goal nuts are already (g59iwyxf) true, no cost for those nuts.
    - Return zero for goal state.

    Note: The heuristic is not guaranteed to be admissible but aims to be informative and efficiently computable.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Build location adjacency graph from static facts (kxokb5hx loc1 loc2).
        self.adj = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "kxokb5hx":
                loc1, loc2 = parts[1], parts[2]
                self.adj.setdefault(loc1, set()).add(loc2)
                self.adj.setdefault(loc2, set()).add(loc1)

        # Extract nuts that should be achieved in the goal (g59iwyxf nut).
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "g59iwyxf":
                self.goal_nuts.add(parts[1])

    def __call__(self, node):
        state = node.state

        # Quick check if goal is reached
        if self.goals <= state:
            return 0

        # Helper to get current location of an object based on (u4srx1nb obj loc).
        # Objects can be nuts, spanners or wrenches (ilfxqhcc).
        obj_to_loc = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "u4srx1nb" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                obj_to_loc[obj] = loc

        # Find all wrenches (ilfxqhcc) locations: those that appear as first argument of (u4srx1nb ?wrench ?loc)
        # but ?wrench type check omitted; we just trust that u4srx1nb ?ilfxqhcc ?loc means wrench location.
        wrenches = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "u4srx1nb":
                obj = parts[1]
                # If this fact's obj is not a known spanner or nut, identify as wrench candidate.
                # But safer: gather all objects appearing in u4srx1nb, and heuristic focuses on those also in yud7t3vb (wrenches linked to spanners).
                # Hence, we use wrenches from yud7t3vb predicates.
        # Extract all wrenches from yud7t3vb facts: (yud7t3vb wrench spanner)
        wrench_to_spanner = {}
        spanner_to_wrenches = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "yud7t3vb":
                wrench, spanner = parts[1], parts[2]
                wrench_to_spanner[wrench] = spanner
                spanner_to_wrenches.setdefault(spanner, set()).add(wrench)
                wrenches.add(wrench)

        # For wrenches, get their current locations
        wrench_locs = {}
        for wrench in wrenches:
            if wrench in obj_to_loc:
                wrench_locs[wrench] = obj_to_loc[wrench]

        # Identify which spanners are ready for use (g0brvc06 spanner)
        ready_spanners = {parts[1] for parts in (get_parts(f) for f in state) if parts[0] == "g0brvc06"}

        # Identify nuts to achieve (subset of goal nuts not yet satisfied)
        nuts_to_achieve = [nut for nut in self.goal_nuts if ("(g59iwyxf %s)" % nut) not in state]

        # Helper: BFS to find shortest distance between locations in location graph
        def shortest_path_length(start, goal):
            if start == goal:
                return 0
            visited = {start}
            frontier = [(start, 0)]
            while frontier:
                current, dist = frontier.pop(0)
                for nbr in self.adj.get(current, []):
                    if nbr == goal:
                        return dist + 1
                    if nbr not in visited:
                        visited.add(nbr)
                        frontier.append((nbr, dist + 1))
            return None  # No path found

        total_cost = 0

        for nut in nuts_to_achieve:
            # Current location of the nut
            nut_loc = obj_to_loc.get(nut)
            if nut_loc is None:
                # Nut location unknown: pessimistically assume high cost
                total_cost += 10
                continue

            # Find a spanner that is ready and associated with a wrench we can move
            # If no spanner associated with wrench is ready, will still try heuristically
            spanner_to_use = None
            wrench_of_spanner = None
            min_move_cost = None

            # Consider all ready spanners and their wrenches to find minimal wrench movement
            for spanner in ready_spanners:
                wrenches_for_spanner = spanner_to_wrenches.get(spanner, set())
                for wrench in wrenches_for_spanner:
                    loc = wrench_locs.get(wrench)
                    if loc is None:
                        continue
                    dist = shortest_path_length(loc, nut_loc)
                    if dist is None:
                        continue
                    # +1 action for wrench activation on spanner (moi6wnfu)
                    # +1 action for unlocking the nut (kjsqfni8)
                    est_cost = dist + 2
                    if min_move_cost is None or est_cost < min_move_cost:
                        min_move_cost = est_cost
                        spanner_to_use = spanner
                        wrench_of_spanner = wrench

            if min_move_cost is None:
                # No reachable spanner/wrench found; penalize heuristic but keep finite
                min_move_cost = 10

            total_cost += min_move_cost

        return total_cost
