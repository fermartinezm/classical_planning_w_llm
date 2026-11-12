from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
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
    A domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal state by focusing on:
    (a) The number of nuts (rupjajag) that have not yet been processed to the final goal predicate (g59iwyxf).
    (b) The minimal estimated number of moves needed to transport tools (ilfxqhcc) between locations (i6azntdy).
    (c) The status of spanners (e2zu8y93) that must be "deactivated" (i.e., g0brvc06 false) to enable processing of nuts.
    The heuristic approximates the effort of moving tools to locations to enable processing of nuts until all are in the goal condition.

    Assumptions
    -----------
    - The goal requires all `(g59iwyxf ?nut)` predicates true for all nuts in goal.
    - Nuts start at certain locations and must be processed via spanners/tools/tools location changes.
    - The tool (ilfxqhcc) must be moved along a chain of locations (connected by kxokb5hx predicates).
    - Spanners `g0brvc06` indicate if a spanner is ready to be used; it can be turned off by the kjsqfni8 action which also produces the goal predicate for nuts.
    - The heuristic treats moving a tool one step along a kxokb5hx edge as one action (n9sbfmrs).
    - The heuristic counts:
      * The number of nuts still needing processing.
      * The distance from the current tool location to the nut's location.
      * The number of spanners that need processing.

    Heuristic Initialization
    -------------------------
    - Extract all static edges `(kxokb5hx ?from ?to)` defining a directed graph of locations.
    - Extract all nuts that appear in the goal `(g59iwyxf ?nut)`.
    - Precompute and store adjacency for shortest path calculations.
    - Store all spanners and their states.
    - Store initial tool locations (u4srx1nb predicate) per tool.

    Step-By-Step Thinking for Computing Heuristic
    ----------------------------------------------
    1. Identify nuts that have not reached the goal yet (no `(g59iwyxf ?nut)` fact).
    2. For each such nut:
       - Identify its current location (from `(u4srx1nb ?nut ?loc)`).
       - Determine if it is “ready” for the final action, i.e., if corresponding spanner and tool conditions are met.
    3. For tools (ilfxqhcc), find their current locations.
    4. Compute shortest paths between locations using the adjacency graph.
    5. For each tool, estimate the minimal number of moves needed to bring it to either:
       - The location of a nut to enable processing, or
       - The location of a spanner that can be used to disable the spanner for nut processing.
    6. Count:
       - For each nut not processed, at least one `kjsqfni8` action is required.
       - For each tool movement along locations, estimate the number of `n9sbfmrs` moves needed.
       - For each spanner needing to be processed (turned off), add a step.
    7. Sum these costs as a heuristic estimation of action count needed to achieve the goal.
    8. Return 0 if all nuts are already in goal state (goal reached).

    The heuristic is not admissible but combines distance and counts of processing requirements to guide greedy best-first search efficiently.
    """

    def __init__(self, task):
        # Goal nuts are those nuts for which g59iwyxf must be true
        self.goal_nuts = set()
        for g in task.goals:
            p, *args = get_parts(g)
            if p == "g59iwyxf" and len(args) == 1:
                self.goal_nuts.add(args[0])  # nut identifier

        # Store adjacency: kxokb5hx edges represent directed edges between locations
        self.adjacency = {}
        for fact in task.static:
            if match(fact, "kxokb5hx", "*", "*"):
                _, from_loc, to_loc = get_parts(fact)
                self.adjacency.setdefault(from_loc, set()).add(to_loc)

        # All locations known from adjacency keys and values
        self.locations = set(self.adjacency.keys())
        for targets in self.adjacency.values():
            self.locations.update(targets)

        # Precompute shortest path lengths between all pairs of locations using BFS
        # As the graph appears small, BFS on demand is efficient enough.
        # Store adjacency for BFS
        self.adj = self.adjacency  # alias
        # Store tool objects and spanner objects for quick reference
        self.tools = set()
        self.spanners = set()
        for fact in task.facts:
            p, *args = get_parts(fact)
            if p == "ilfxqhcc":  # type of tool objects (though not facts, used for completeness)
                # Actually types are not stored as facts; so tools identified by domain objects of this type might not be accessible.
                pass
        # Instead, we identify tools and spanners dynamically from facts during heuristic call.

        # Store the initial predicate names from task for potential use if needed
        self.all_tools = set()
        self.all_spanners = set()
        for fact in task.initial_state | task.static | task.goals:
            # Gather objects that are ilfxqhcc or e2zu8y93 from facts keys
            p, *args = get_parts(fact)
            if p == "yud7t3vb" or p == "u4srx1nb":
                for arg in args:
                    # We will identify tools and spanners dynamically in __call__
                    pass

        # We'll identify spanners and tools in __call__ using types seen in state.

    def shortest_distance(self, start, goal):
        """Breadth-first search to compute shortest path length from start to goal in `self.adj` graph."""
        if start == goal:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            neighbors = self.adj.get(current, set())
            for n in neighbors:
                if n == goal:
                    return dist + 1
                if n not in visited:
                    visited.add(n)
                    queue.append((n, dist + 1))
        # No path found, return a large number
        return 1000

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is zero
        if all(g in state for g in self.goal_nuts_to_facts()):
            return 0

        # Identify all nuts in the problem from goal nuts and current predicates
        # Nuts have type rupjajag - identify nuts that appear in any related fact (u4srx1nb or g59iwyxf)
        nuts_in_state = set()
        for fact in state:
            p, *args = get_parts(fact)
            if p in ["u4srx1nb", "g59iwyxf", "bu1om0m1"]:
                for arg in args:
                    if arg.startswith("nut"):
                        nuts_in_state.add(arg)
        # But rely mainly on goal nuts as identification set
        nuts = self.goal_nuts

        # Identify nuts already achieved (g59iwyxf predicate)
        nuts_done = {n for n in nuts if f"(g59iwyxf {n})" in state}
        nuts_pending = nuts - nuts_done

        # If all nuts done, heuristic is zero
        if len(nuts_pending) == 0:
            return 0

        # Identify spanners (e2zu8y93 type) and their states:
        # (g0brvc06 ?spanner) means spanner active (needs disabling)
        # We identify all spanner objects currently active
        spanners_active = set()
        spanner_objects = set()
        for fact in state:
            p, *args = get_parts(fact)
            if p == "g0brvc06" and len(args) == 1:
                spanners_active.add(args[0])
            if p == "yud7t3vb" and len(args) >= 1:
                spanner_objects.add(args[1])
        # Also spanners that appear in yud7t3vb predicate for tool-to-spanner relation
        for fact in state:
            p, *args = get_parts(fact)
            if p == "yud7t3vb" and len(args) >= 2:
                spanner_objects.add(args[1])

        # Identify tools (ilfxqhcc) and their current locations:
        # (u4srx1nb ?tool ?loc) for tool location
        tools_locations = {}
        tools_set = set()
        for fact in state:
            p, *args = get_parts(fact)
            if p == "u4srx1nb" and len(args) == 2:
                obj, loc = args
                # Identify if obj is a tool (type ilfxqhcc) by presence in yud7t3vb or goal predicates
                # we use heuristic: tools appear in yud7t3vb predicate (tool spanner)
                # In domain, ?v7he5 - ilfxqhcc is param in yud7t3vb
                # So we identify tools by appearance in yud7t3vb predicates' first argument
                # Let's gather all tools by scanning yud7t3vb predicates
                pass
        # So we collect all tools from yud7t3vb facts
        tools_set = set()
        tool_to_spanner = {}
        for fact in state:
            p, *args = get_parts(fact)
            if p == "yud7t3vb" and len(args) == 2:
                tool, spanner = args
                tools_set.add(tool)
                tool_to_spanner[tool] = spanner

        # Now collect tool locations from u4srx1nb for all tools found
        for fact in state:
            p, *args = get_parts(fact)
            if p == "u4srx1nb" and len(args) == 2:
                obj, loc = args
                if obj in tools_set:
                    tools_locations[obj] = loc

        # Identify nut locations from u4srx1nb (nuts are rupjajag)
        nut_locations = {}
        for nut in nuts_pending:
            for fact in state:
                if match(fact, "u4srx1nb", nut, "*"):
                    _, _, loc = get_parts(fact)
                    nut_locations[nut] = loc
                    break
            else:
                # Nut location unknown
                nut_locations[nut] = None

        # Count how many nuts must be processed
        nuts_count = len(nuts_pending)

        # Count how many spanners still active must be disabled
        spanners_count = len(spanners_active)

        # We estimate cost as sum of:
        # - For each nut, one kjsqfni8 action to finalize it
        # - For each spanner active, one kjsqfni8 action is needed to disable it (which also disables g0brvc06 and enables g59iwyxf)
        # - For each tool, minimal moves to get tool to spanner location (yud7t3vb pairs tool+spanner strongly linked)
        # - For each nut, minimal moves to tool location next to nut

        # Compute minimal tool moves:
        # For each tool, minimal distance from its current location to the spanner it must affect,
        # plus from spanner location to nut location if needed.

        # First get spanner locations from (yud7t3vb ?tool ?spanner) and (u4srx1nb ?spanner ?loc)
        spanner_locations = {}
        for spanner in spanner_objects:
            for fact in state:
                if match(fact, "u4srx1nb", spanner, "*"):
                    _, sp, loc = get_parts(fact)
                    spanner_locations[spanner] = loc
                    break

        # Total tool move cost: sum over tools of distance from tool location to its spanner location
        total_tool_moves = 0
        for tool, spanner in tool_to_spanner.items():
            tool_loc = tools_locations.get(tool, None)
            spanner_loc = spanner_locations.get(spanner, None)
            if tool_loc is None or spanner_loc is None:
                continue
            dist = self.shortest_distance(tool_loc, spanner_loc)
            total_tool_moves += dist

        # Also count minimal moves from spanner location to nut location for nuts related to the spanner's tool
        # We approximate nut processing cost by number of nuts (each needs at least one kjsqfni8 action)

        # For nuts associated with spanners that are not yet disabled,
        # assume cost of at least one move to position tool and one action to disable spanner.

        # Overall cost estimation:
        # nuts_count (each needs one kjsqfni8)
        # + spanners_count (each must be disabled by kjsqfni8)
        # + total_tool_moves (moves needed to bring tools to spanners)
        # + minimal moves to bring tool/spanner combination close to nut location

        # Estimate movements tool -> spanner -> nut
        tool_to_nut_moves = 0

        # For each nut, find minimal distance from related spanner location to nut location
        # We first find spanner related to each tool, and for each nut, approximate using closest spanner
        for nut in nuts_pending:
            nut_loc = nut_locations.get(nut)
            if nut_loc is None:
                continue

            min_dist_to_nut = 1000
            # For spanners active or all spanners (since they relate to a tool)
            for spanner, sp_loc in spanner_locations.items():
                dist = self.shortest_distance(sp_loc, nut_loc)
                if dist < min_dist_to_nut:
                    min_dist_to_nut = dist
            tool_to_nut_moves += min_dist_to_nut

        # Combine all estimated costs
        estimate = nuts_count + spanners_count + total_tool_moves + tool_to_nut_moves

        # Cap heuristic to at least 1 if estimate is zero but goal not reached, to avoid zero in non-goal states
        if estimate == 0:
            estimate = 1

        return estimate

    def goal_nuts_to_facts(self):
        """Convert goal nuts set to corresponding goal facts format."""
        return {f"(g59iwyxf {nut})" for nut in self.goal_nuts}
