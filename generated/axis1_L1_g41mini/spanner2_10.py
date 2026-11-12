from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(u4srx1nb bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (domain e2zu8y93).

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal state,
    where the goal consists of achieving `(g59iwyxf ?nut)` for all `nut` objects (rupjajag type).
    The key goal is to transform `(bu1om0m1 ?nut)` facts into `(g59iwyxf ?nut)` by applying a sequence of
    three types of actions that move objects `v7he5` (ilfxqhcc), spanners (e2zu8y93), and nuts (rupjajag)
    across locations (i6azntdy).

    The heuristic leverages the following observations from the domain and example instances:
    - Each `nut` that has `(bu1om0m1 ?nut)` true but not `(g59iwyxf ?nut)` must be "activated".
    - Activation requires:
      1. Moving a character `v7he5` (ilfxqhcc) along `i6azntdy` positions via `n9sbfmrs` actions,
         because `(u4srx1nb ?v7he5 ?location)` changes only via these moves along connected locations (`kxokb5hx`).
      2. Then using that character and a spanner with `(g0brvc06 ?spanner)` and `(yud7t3vb ?v7he5 ?spanner)` facts,
         the action `moi6wnfu` disables `(u4srx1nb ?aqz3v ?location)` for the spanner, enabling `(yud7t3vb ?v7he5 ?aqz3v)`.
      3. Finally, the `kjsqfni8` action converts `(bu1om0m1 ?nut)` into `(g59iwyxf ?nut)` if all prerequisites are met.
    - Consequently, a nut not yet activated requires:
      - Getting its "user" (ilfxqhcc) to the nut's location (or a nearby connected location).
      - Using the appropriate spanner on that user.
      - Executing `kjsqfni8` to finalize the nut activation.
    - The main cost is travelling for the user on the `kxokb5hx` graph and performing the three action types.

    Assumptions
    -----------
    - Only one user (ilfxqhcc) is relevant for each nut (from `(u4srx1nb ?v7he5 ?location)` and `(yud7t3vb ?v7he5 ?spanner)`).
    - The domain's spatial connectivity `kxokb5hx` forms a graph to compute shortest paths on.
    - Costs of moving users along locations are approximated by shortest paths on that graph.
    - Moving a user from current to nut's location costs approximately the shortest path length (number of `n9sbfmrs` actions).
    - The `moi6wnfu` and `kjsqfni8` actions each count as one step per nut.
    - We ignore the order and concurrency (heuristic can overestimate but should guide search efficiently).
    - The heuristic returns 0 if the goal `(g59iwyxf nut)` is achieved for all nuts.
    - The heuristic always returns finite non-negative integers for solvable states.

    Heuristic Initialization
    -------------------------
    - Extract all static location-to-location connections `(kxokb5hx ?loc1 ?loc2)` 
      and build an undirected graph for user movement.
    - Extract the goal nuts that must satisfy `(g59iwyxf ?nut)`.
    - Extract ilfxqhcc characters and their current locations `(u4srx1nb ?v7he5 ?loc)`.
    - Extract the positions of all nuts `(u4srx1nb ?nut ?loc)` that are `rupjajag`.
    - Extract which nuts are already activated by checking presence of `(g59iwyxf ?nut)`.

    Step-By-Step Thinking for Computing Heuristic
    ----------------------------------------------
    1. If all `(g59iwyxf ?nut)` goals hold true, return 0.
    2. Otherwise, repeat for each nut that:
       - Has `(bu1om0m1 ?nut)` (not yet activated).
       - Needs to be activated via:
         a) Moving a user (ilfxqhcc) from current position to the nut's location or a neighboring connected location.
            We approximate this by the minimal shortest path distance for any user to the nut location.
         b) Executing one `moi6wnfu` action (use spanner).
         c) Executing one `kjsqfni8` action (activate nut).
       The user movement cost counts as path length.
    3. Sum these minimal costs over all nuts needing activation.
    4. Return the total as the heuristic value.

    This provides a fast-to-compute but informed heuristic estimating how far the user characters
    are from nuts and how many actions remain to activate them.
    """

    def __init__(self, task):
        """Extract static graph connectivity, nuts and user locations from the static facts and initial state."""
        self.goals = task.goals
        self.static = task.static

        # Extract all static location connectivity as an undirected graph.
        self.graph = dict()  # loc -> set of connected locs
        for fact in self.static:
            if match(fact, "kxokb5hx", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:]
                self.graph.setdefault(loc1, set()).add(loc2)
                self.graph.setdefault(loc2, set()).add(loc1)

        # Precompute the set of nuts to activate: nuts in the goal requiring g59iwyxf
        self.nuts_to_activate = set()
        for goal in self.goals:
            if match(goal, "g59iwyxf", "*"):
                nut = get_parts(goal)[1]
                self.nuts_to_activate.add(nut)

    def shortest_path_length(self, start, goal):
        """Compute shortest path length in undirected graph between start and goal locations using BFS."""
        if start == goal:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            for neighbor in self.graph.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        # No path found
        return None

    def __call__(self, node):
        """Compute the heuristic value for the given search node."""
        state = node.state

        # Quick check if all goal nuts are activated
        all_activated = True
        for nut in self.nuts_to_activate:
            if f"(g59iwyxf {nut})" not in state:
                all_activated = False
                break
        if all_activated:
            return 0

        # Extract current user (ilfxqhcc) locations
        user_locations = set()
        for fact in state:
            if match(fact, "u4srx1nb", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # ilfxqhcc type objects have a naming pattern unknown,
                # but we assume from domain and instances they appear also as 'bob', 'v7he5'-type
                # We check if obj is ilfxqhcc by types from domain file is not possible here,
                # so we gather all with usage below.
                # We assume that objects appearing in yud7t3vb as first parameter are ilfxqhcc users.
                # Hence we gather all ilfxqhcc by scanning yud7t3vb facts.
                # To simplify, we collect all candidates here as users are few:
                # We will filter properly below.
                user_locations.add((obj, loc))

        # Extract known ilfxqhcc users by checking who has yud7t3vb facts
        users = set()
        for fact in state:
            if match(fact, "yud7t3vb", "*", "*"):
                user = get_parts(fact)[1]
                users.add(user)

        # Filter user_locations for actual users from yud7t3vb facts
        user_locations = {loc for (obj, loc) in user_locations if obj in users}

        # Extract nuts positions and which are active
        nut_locations = dict()  # nut -> location of nut (where u4srx1nb nut location)
        nuts_bu1om0m1 = set()   # nuts with (bu1om0m1 ?nut) true (need activation)
        for fact in state:
            if match(fact, "u4srx1nb", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in self.nuts_to_activate:
                    nut_locations[obj] = loc
            if match(fact, "bu1om0m1", "*"):
                nut = get_parts(fact)[1]
                nuts_bu1om0m1.add(nut)

        # Nuts requiring activation are those that have bu1om0m1 true and g59iwyxf false (not activated)
        nuts_to_activate = [nut for nut in self.nuts_to_activate if (nut in nuts_bu1om0m1 and f"(g59iwyxf {nut})" not in state)]

        # If no users or no nuts to activate, heuristic is large but finite
        # Return length estimate as large number in that edge case
        if not user_locations or not nuts_to_activate:
            return len(nuts_to_activate) * 5  # rough upper bound

        total_cost = 0
        # For each nut to activate, estimate cost: 
        # minimal user move cost + 2 (one moi6wnfu + one kjsqfni8)
        # For user move: shortest path from any user location to nut location on graph

        for nut in nuts_to_activate:
            nut_loc = nut_locations.get(nut, None)
            if nut_loc is None:
                # If nut location unknown, use default cost 5
                total_cost += 5
                continue
            best_dist = None
            # Try all user current locations
            for user_loc in user_locations:
                dist = self.shortest_path_length(user_loc, nut_loc)
                if dist is not None and (best_dist is None or dist < best_dist):
                    best_dist = dist
            if best_dist is None:
                # No path found for this nut, heuristic high value:
                best_dist = 10
            # Add cost: user moves + 1 moi6wnfu + 1 kjsqfni8
            total_cost += best_dist + 2

        return total_cost
