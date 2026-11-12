from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(qkdx0xe2 obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain (du770mtw).

    Summary:
        This heuristic estimates the minimum number of actions required to achieve the goal conditions from a given state.
        The goal conditions are of the form (ne6sl557 ?ris62) on wqid8mog objects.
        Achieving a goal fact (ne6sl557 ?obj) requires firing the action f7y6nzq3 involving the object, a du770mtw object with (ndg963qz),
        a pt4b08k4 object with fluent (fqyrzl30), and a shared mz6r1d30 location among involved objects.
        The heuristic thus focuses on estimating the cost to "activate" each goal object by reaching required intermediate conditions.

    Assumptions:
    - Each goal (ne6sl557 ?r) corresponds exactly to one unique wqid8mog object.
    - The central "activation" involves the target object, a du770mtw object that must have (ndg963qz),
      and a pt4b08k4 object that must be "associated" (fqyrzl30) with that du770mtw object.
    - The location relationships are encoded by qkdx0xe2 predicates linking pt4b08k4 or wqid8mog or du770mtw objects to mz6r1d30 locations.
    - The lpmls56r relation defines a static graph of mz6r1d30 locations for movement.
    - The main challenge is to move qkdx0xe2 facts to satisfy preconditions of f7y6nzq3, requiring qkdx0xe2 facts and fqyrzl30 fluents, plus ndg963qz for involved du770mtw objects.
    - Actions jql8txb3 and nm6lzp2x enable moving/fixing these needed predicates.
    - The heuristic ignores incomplete combinations and estimates minimal "distance" in the location graph between involved objects and the required intermediate facts.

    Heuristic Initialization:
    - Extract all static facts (lpmls56r): build an adjacency map defining the undirected graph of mz6r1d30 locations.
    - Determine all du770mtw objects that currently have (ndg963qz) predicate in the initial state and note their locations.
    - The goal is to obtain (ne6sl557 ?ris62) for goal wqid8mog objects.
    - The heuristic will:
       - Identify how many goal objects are not yet "activated" (ne6sl557 missing).
       - For each such goal object, estimate minimal number of steps to satisfy f7y6nzq3 preconditions, primarily by:
         - Estimating minimal path distances in the static location graph for required qkdx0xe2 predicates to be on shared mz6r1d30 locations.
         - Counting the minimal number of required jql8txb3 and nm6lzp2x actions to rearrange qkdx0xe2 and fqyrzl30 predicates.
         - Including the final f7y6nzq3 action itself.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each goal (ne6sl557 ?obj), if it already holds in the state, cost contribution is zero.
    2. For each goal object not yet activated:
        a. Identify its current location(s) as indicated by (qkdx0xe2 ?obj ?location).
           If multiple locations, pick the first or all to minimize distance.
        b. Identify all du770mtw objects in the state with (ndg963qz) predicate; note their locations (from qkdx0xe2).
        c. Identify pt4b08k4 objects that have the (fqyrzl30 ?pt4b08k4 ?du770mtw) fluent.
        d. For each combination of:
           - a pt4b08k4 object with (fqyrzl30 pt4b08k4 du770mtw_obj),
           - the du770mtw_obj having (ndg963qz),
           compute the minimal distance in location graph so that:
             - The goal object, pt4b08k4 object, and wqid8mog object are connected via correct qkdx0xe2 predicates on a shared or reachable mz6r1d30 location.
        e. Estimate cost as number of moves (nm6lzp2x) to move qkdx0xe2 predicates closer plus changes in fqyrzl30 fluents (jql8txb3) plus final f7y6nzq3.
    3. Choose minimal cost among possible combinations for the given goal object.
    4. Sum these minimal costs for all goal objects not yet achieved.
    5. Return the sum as heuristic cost.

    Notes:
    - Since the domain names are obfuscated, the heuristic cannot rely on semantic names and must use structural relationships.
    - The heuristic approximates the minimal number of "move" and "activate" actions by estimating shortest paths in lpmls56r graph and counting required predicate/fact changes.
    - Final heuristic value is zero only if all goals are satisfied.
    - The heuristic ignores possible concurrency or parallel actions for simplicity.
    """

    def __init__(self, task):
        """Initialize by processing static facts and indexing objects by predicate."""

        self.goals = task.goals  # The goal facts (ne6sl557 ?obj) requiring target objects to be activated.

        static_facts = task.static

        # Extract static undirected graph of mz6r1d30 locations from lpmls56r predicate.
        # lpmls56r loc1 loc2  means edge loc1 <-> loc2
        self.location_graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "lpmls56r":
                loc1, loc2 = parts[1], parts[2]
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

        # Precompute adjacency for pathfinding
        # Since graph is static and likely small, we will do BFS from any node as needed.

    def shortest_path_length(self, start, goal):
        """Compute shortest path length between two locations in the location graph using BFS."""
        if start == goal:
            return 0
        visited = set()
        frontier = [(start, 0)]
        while frontier:
            current, dist = frontier.pop(0)
            if current == goal:
                return dist
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.location_graph.get(current, ()):
                if neighbor not in visited:
                    frontier.append((neighbor, dist + 1))
        # If no path, return a large penalty distance
        return 1000

    def __call__(self, node):
        """
        Compute heuristic cost for the given state.
        Steps:
         - Identify goal wqid8mog objects missing ne6sl557 (activated) predicate.
         - For each missing:
            - Determine current mz6r1d30 location(s) of the wqid8mog object via (qkdx0xe2).
            - Determine du770mtw objects with (ndg963qz); get their locations.
            - Determine pt4b08k4 objects with (fqyrzl30 pt4b08k4 du770mtw_obj).
            - Identify qkdx0xe2 facts of pt4b08k4 and du770mtw objects.
            - Estimate minimal combined cost to align required predicates on some location to enable f7y6nzq3 action.
            - Add 1 for the final f7y6nzq3 action.
         - Sum all per-goal estimates.
        """

        state = node.state

        # Quick check: if all goals achieved, heuristic = 0.
        if self.goals <= state:
            return 0

        # Index relevant predicates from current state:
        # qkdx0xe2 ?obj ?loc
        qkdx0xe2_map = {}  # obj -> set of locations
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "qkdx0xe2" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                qkdx0xe2_map.setdefault(obj, set()).add(loc)

        # fqyrzl30 ?pt4b08k4 ?du770mtw
        fqyrzl30_map = {}  # pt4b08k4_obj -> set of du770mtw_obj
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "fqyrzl30" and len(parts) == 3:
                pt_obj, du_obj = parts[1], parts[2]
                fqyrzl30_map.setdefault(pt_obj, set()).add(du_obj)

        # ndg963qz ?du770mtw_obj
        ndg963qz_set = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "ndg963qz" and len(parts) == 2:
                ndg963qz_set.add(parts[1])

        # Gather goal wqid8mog objects needing activation: those in goal with ne6sl557 but missing in current state
        goal_wqid8mog = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "ne6sl557" and len(parts) == 2:
                obj = parts[1]
                if f"(ne6sl557 {obj})" not in state:
                    goal_wqid8mog.append(obj)

        # A helper to find if wqid8mog object has qkdx0xe2 location(s)
        # If none, heuristic is high penalty since object is disconnected.

        total_cost = 0

        for w_obj in goal_wqid8mog:

            # Locations of wqid8mog object for qkdx0xe2
            w_locs = qkdx0xe2_map.get(w_obj, None)
            if not w_locs:
                # No known location, assign high heuristic penalty
                total_cost += 1000
                continue

            min_goal_cost = 10000  # Initialize large cost

            # Try all combinations as:
            # For each du770mtw_obj with ndg963qz, get its location(s)
            # For each pt4b08k4_obj with fqyrzl30(pt4b08k4_obj, du770mtw_obj)
            for du_obj in ndg963qz_set:
                du_locs = qkdx0xe2_map.get(du_obj, None)
                if not du_locs:
                    continue
                for pt_obj, linked_du_objs in fqyrzl30_map.items():
                    if du_obj not in linked_du_objs:
                        continue
                    pt_locs = qkdx0xe2_map.get(pt_obj, None)
                    if not pt_locs:
                        continue

                    # Now we have:
                    # wobj locations (w_locs), pt4b08k4 locations (pt_locs), du770mtw locations (du_locs)

                    # Heuristic idea:
                    # Estimate cost to bring all three predicates (qkdx0xe2 for w_obj, pt_obj, du_obj)
                    # to the same location or within reachable locations to satisfy f7y6nzq3 preconditions.

                    # Minimal distance to connect these three location sets by minimal sum of pairwise shortest path distances.

                    # We try all combination triple loc_w, loc_pt, loc_du to find minimal sum of:
                    # dist(loc_w, loc_pt) + dist(loc_pt, loc_du) + dist(loc_du, loc_w)
                    # Since the effective position would be the meeting point they can share.

                    # More simply, we can approximate by:
                    # Find minimal sum of distances to a meeting point location.

                    # To reduce complexity:
                    # Find all locations in the graph, then for each location compute sum of distance to loc_w, loc_pt and loc_du.
                    # Pick minimal over all meeting points.

                    candidate_meeting_locs = set(self.location_graph.keys())

                    # For performance limit search to union of w_locs, pt_locs, du_locs plus their neighbors
                    # But since domains are small, we can use full location_graph keys.

                    min_meeting_cost = 10000

                    for meet_loc in candidate_meeting_locs:
                        min_dist_w = min(self.shortest_path_length(w_loc, meet_loc) for w_loc in w_locs)
                        min_dist_pt = min(self.shortest_path_length(pt_loc, meet_loc) for pt_loc in pt_locs)
                        min_dist_du = min(self.shortest_path_length(du_loc, meet_loc) for du_loc in du_locs)

                        # The minimum total moves required to bring all to meet_loc
                        total_moves = min_dist_w + min_dist_pt + min_dist_du

                        if total_moves < min_meeting_cost:
                            min_meeting_cost = total_moves

                    if min_meeting_cost < 10000:
                        # Estimated:
                        # - nm6lzp2x actions roughly proportional to number of moves of qkdx0xe2 predicates
                        # - jql8txb3 action (changing fqyrzl30 predicate) needed once to make (fqyrzl30 ?pt4b08k4 ?du770mtw)
                        #   but as the fluent is true in current state, assume no cost for jql8txb3 here.
                        # - Finally, 1 f7y6nzq3 action needed to achieve (ne6sl557 ?wqid8mog)
                        # Hence heuristic cost = moves + 1 final action

                        # Add estimated moves plus 1 final action firing
                        est_cost = min_meeting_cost + 1

                        if est_cost < min_goal_cost:
                            min_goal_cost = est_cost

            # If no combination found, fallback penalty cost
            if min_goal_cost == 10000:
                min_goal_cost = 1000

            total_cost += min_goal_cost

        return total_cost
