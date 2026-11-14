from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(u4srx1nb bob shed)".
    - `args`: The expected pattern elements (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (domain e2zu8y93).

    Summary
    -------
    This heuristic estimates the number of required actions to achieve all goals
    `(g59iwyxf ?nut)` for all nuts in the goal. The ultimate goal is to have all 
    nuts satisfy `g59iwyxf`. The heuristic estimates how many actions are needed 
    to convert all nuts that currently have the predicate `(bu1om0m1 ?nut)` (meaning 
    they are still "unprocessed") into having `(g59iwyxf ?nut)`.

    Assumptions
    -----------
    - Nuts start in some location and can be moved with tools called spanners.
    - Spanners have a boolean predicate `(g0brvc06 ?spanner)` which indicates
      whether they are currently "usable".
    - Nuts start with `(bu1om0m1 ?nut)` indicating they still need to be processed.
    - A nut is considered "processed" when `(g59iwyxf ?nut)` is true.
    - The domain actions:
        * `n9sbfmrs` moves ilfxqhcc (?v7he5) from one location to another along connections (`kxokb5hx`).
        * `moi6wnfu` attaches ilfxqhcc (?v7he5) to e2zu8y93 (?aqz3v), removing presence of `u4srx1nb ?aqz3v ?r2bzp`.
        * `kjsqfni8` processes the nut (?gdh2s) given correct preconditions, removing `(bu1om0m1 ?gdh2s)` and `(g0brvc06 ?aqz3v)`, adding `(g59iwyxf ?gdh2s)`.
    - Actions require at least one spanner and one location in common; spanners may need moving.
    - The graph of locations given by `(kxokb5hx ?from ?to)` is effectively the connectivity graph to move objects.

    Heuristic Initialization
    ------------------------
    - From static facts, build a bidirectional graph of locations (using `(kxokb5hx)`).
    - Extract goal nuts that require `(g59iwyxf ?nut)`.
    - Extract nuts that must be processed `(bu1om0m1 ?nut)`.
    - Extract spanners and their availability `(g0brvc06 ?spanner)`.
    - Store initial locations of objects relevant for estimating distance.
    - Extract connections to enable shortest path computations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify nuts still unprocessed (with `(bu1om0m1 ?nut)` and NOT `(g59iwyxf ?nut)`).
    2. Identify all spanners and if they are currently available `(g0brvc06)`.
    3. For each nut still unprocessed:
       a. Identify the current location of the nut `(u4srx1nb ?nut ?loc)`; nuts are at locations.
       b. Identify a spanner location and availability.
    4. For each nut, estimate:
       - The minimal travel distance for the spanner to get to the nut's location, plus
       - One move of the spanner to the nut's location (if not already there), plus
       - One action `kjsqfni8` to process the nut.
    5. Since each `kjsqfni8` removes `(bu1om0m1)` and adds `(g59iwyxf)`, we count one action per nut for this processing.
    6. Additionally, we need to estimate the minimal number of moves to bring spanners to nuts, favoring spanners already at the nut's location.
    7. Spanner moves (`n9sbfmrs` actions) are estimated using shortest path distances on the graph.
    8. Moving the "ilfxqhcc" (?v7he5) (likely an agent) to operate the spanner is not explicitly needed because `kjsqfni8` requires spanner and nut at same location.
    9. The `moi6wnfu` action (attaching ilfxqhcc to e2zu8y93) is not accounted explicitly because the heuristic focuses on processing nuts.
    10. Summing across all nuts the minimal spanner travel + 1 for processing provides a heuristic estimate.
    11. If no spanners are available (no `(g0brvc06 ?spanner)`), estimate a penalty for re-enabling one spanner (requires 1 `kjsqfni8` on some spanner? but domain doesn't say how to re-enable; ignoring this rare case; heuristic returns a high cost).

    The heuristic is not admissible but attempts to reflect the minimal necessary spanner repositioning plus one processing per nut.

    Implementation details:
    - We precompute shortest paths with BFS on the locations graph.
    - Use the heuristic value = sum over nuts (distance (nearest spanner location to nut location) + 1 (processing)).
    - If goal reached, heuristic = 0.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Build bidirectional graph of locations based on (kxokb5hx ?from ?to)
        self.location_graph = {}
        for fact in self.static:
            if match(fact, "kxokb5hx", "*", "*"):
                parts = get_parts(fact)
                loc_from, loc_to = parts[1], parts[2]
                self.location_graph.setdefault(loc_from, set()).add(loc_to)
                self.location_graph.setdefault(loc_to, set()).add(loc_from)

        # Extract nuts that appear in goal as (g59iwyxf ?nut)
        self.goal_nuts = set()
        for goal_fact in self.goals:
            if match(goal_fact, "g59iwyxf", "*"):
                nut = get_parts(goal_fact)[1]
                self.goal_nuts.add(nut)

        # Precompute all pairwise shortest distances (for efficiency only between needed nodes)
        # We will run BFS on demand rather than all pairs because the graph can be large.
        # Instead, we prepare a function for shortest path distance.

    def shortest_distance(self, start, goal):
        """
        Compute shortest path distance between two locations in the location graph.
        Uses BFS and returns number of edges on the shortest path.

        Return large number if no path found.
        """
        if start == goal:
            return 0
        frontier = [start]
        explored = {start}
        distance = 0
        while frontier:
            distance += 1
            next_frontier = []
            for node in frontier:
                for neighbor in self.location_graph.get(node, []):
                    if neighbor == goal:
                        return distance
                    if neighbor not in explored:
                        explored.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
        return 10**9  # Effectively infinite if no path exists

    def __call__(self, node):
        state = node.state

        if self.goals <= state:
            # Goal reached
            return 0

        # Facts sets for efficient membership tests
        facts = state

        # Identify spanners and their availability
        spanners = set()
        available_spanners = set()
        spanner_locations = {}
        for fact in facts:
            # (u4srx1nb ?v7he5 ?location)
            if match(fact, "u4srx1nb", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                # Identify if obj is a spanner (type e2zu8y93)
                # We cannot directly distinguish type here; heuristic assumes spanners appear with (g0brvc06 ?obj)
                # So collect all objects with (g0brvc06 ?obj)
                # Will cross-link later
                spanner_locations[obj] = loc

            if match(fact, "g0brvc06", "*"):
                spanner = get_parts(fact)[1]
                available_spanners.add(spanner)
                spanners.add(spanner)

        # Sanity: all available spanners are also spanners
        spanners.update(available_spanners)

        if not spanners:
            # No spanners at all => heuristic returns large number to indicate unsolvability or cost
            return 1000

        # Identify nuts still unprocessed = have (bu1om0m1 nut) but NOT (g59iwyxf nut)
        nuts_to_process = set()
        # Prebuild sets for fast check
        processed_nuts = set()
        nuts_with_bu1om0m1 = set()

        for fact in facts:
            if match(fact, "bu1om0m1", "*"):
                nut = get_parts(fact)[1]
                nuts_with_bu1om0m1.add(nut)
            if match(fact, "g59iwyxf", "*"):
                nut = get_parts(fact)[1]
                processed_nuts.add(nut)

        # The nuts of interest must appear in goal nuts as well (since goals require processed nuts)
        for nut in self.goal_nuts:
            if nut in nuts_with_bu1om0m1 and nut not in processed_nuts:
                nuts_to_process.add(nut)

        if not nuts_to_process:
            # Nothing left to do, heuristic = 0
            return 0

        # Determine location of each nut to process
        nut_locations = {}
        for fact in facts:
            if match(fact, "u4srx1nb", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                if obj in nuts_to_process:
                    nut_locations[obj] = loc

        # For spanners without location info, ignore them (heuristic pessimistic)
        spanners_with_loc = [s for s in spanners if s in spanner_locations]
        if not spanners_with_loc:
            # No spanners with known location, penalize heavily
            return 1000

        # We estimate cost as:
        # For each nut:
        #   minimal distance from any available spanner location to nut location
        #   + 1 for the processing action kjsqfni8
        #
        # For unavailable spanners, we still consider their locations as potential helpers,
        # but prefer available spanners.


        # Split spanners into available and unavailable sets with known location
        available_spanner_locs = [spanner_locations[s] for s in spanners_with_loc if s in available_spanners]
        unavailable_spanner_locs = [spanner_locations[s] for s in spanners_with_loc if s not in available_spanners]

        total_cost = 0

        for nut in nuts_to_process:
            nut_loc = nut_locations.get(nut)
            if nut_loc is None:
                # Unknown location for nut => pessimistic estimate
                return 1000

            # Compute minimal distance from any available spanner to nut location
            best_dist = 10**9

            # Prefer available spanners first
            for sp_loc in available_spanner_locs:
                dist = self.shortest_distance(sp_loc, nut_loc)
                if dist < best_dist:
                    best_dist = dist

            if best_dist == 10**9:
                # No available spanner path found, consider unavailable spanners
                for sp_loc in unavailable_spanner_locs:
                    dist = self.shortest_distance(sp_loc, nut_loc)
                    if dist < best_dist:
                        best_dist = dist

            # Heuristic cost for this nut:
            # distance to bring a spanner + 1 processing
            # Moving spanner: each edge requires one action (n9sbfmrs)
            # Processing action kjsqfni8 = 1 action
            # Additionally, if spanner already at nut loc, best_dist=0, so cost=1

            total_cost += best_dist + 1

        return total_cost
