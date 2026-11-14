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
    A domain-dependent heuristic for the secret_domain.

    Summary
    -------
    The heuristic estimates the number of actions required to achieve all goal facts of the form
    (g59iwyxf <nut>) for nuts. Achieving (g59iwyxf nut) requires using the action 'kjsqfni8' which
    consumes a nut that is initially (bu1om0m1 nut) and at a location, after the nut and the tool
    (spanner) are correctly positioned and the tool is "used up" in the process.
    The heuristic approximates costs by accounting for:
    - Moving nuts from their current location to the gate and "processing" them with tools (spanners)
      that must be at certain states.
    - The minimum steps needed to "move" the spanners along the chain of locations from their current
      position to the gate where nuts are processed.
    - Counting the number of nuts that still need to be processed.

    Assumptions
    -----------
    - Goals only require (g59iwyxf nut) facts.
    - Each nut must be "processed" via the 'kjsqfni8' action once, consuming a spanner (which is
      reset from state g0brvc06 to not).
    - Spanners start in some location with predicate (g0brvc06 <spanner>).
    - Nuts start located at some gate or location with predicate (u4srx1nb nut <loc>) and (bu1om0m1 nut)
      marking their initial state.
    - Movement between locations is represented by the symmetric graph predicates (kxokb5hx loc1 loc2).
    - The actions 'n9sbfmrs' move a tool (spanner) from one location to the next.
    - To reach the goal, spanners must be moved so they can process nuts at the gate.
    - The heuristic need not be admissible but should be informative.

    Heuristic Initialization
    -------------------------
    - Extract all nuts from the goals that must be processed ((g59iwyxf nut)).
    - Extract the spanners and their initial state ((g0brvc06 spanner)).
    - Extract the map of locations and their connectivity from (kxokb5hx) to form an undirected graph.
    - Extract initial locations of spanners and nuts from (u4srx1nb) predicates.
    - Identify the gate location(s) where nuts must be processed (utsing (u4srx1nb nut <gate>)).
    - Store static connectivity for shortest path calculations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state already satisfies the goal (all nuts have (g59iwyxf nut)), return 0.

    2. For each nut that does not have (g59iwyxf nut) in the current state, 
       recognize that at least one 'kjsqfni8' action is needed per nut.

    3. For each spanner, find its current location.

    4. For each spanner, compute the shortest distance (number of moves) required to bring it to the 
       nut's location (gate), because spanners must be moved there before using kjsqfni8.

    5. The heuristic sums:
       - The count of nuts yet to be processed (lower bound on kjsqfni8 actions).
       - The minimal total spanner movement cost to bring the spanners to the gate.
         Because spanners can be moved independently along the connectivity graph,
         consider for each unprocessed nut the minimal distance from any spanner to gate,
         then multiply by how many nuts are left (or proportionally approximate).

    6. Add a small constant overhead per nut for required intermediate actions like the 
       'moi6wnfu' actions (which require the spanner and nut at the same location) 
       or spanner tool state resets, approximated here.

    Notes:
    - A shortest path over the location graph is computed using BFS.
    - The heuristic is designed to be efficiently computable by caching shortest paths during the call.
    - This heuristic is not admissible but guides search by encouraging spanners to move efficiently toward nuts to process them.

    """

    def __init__(self, task):
        # All static facts from the task
        static = task.static

        # The set of goal facts (e.g. (g59iwyxf nut))
        self.goal_facts = set(task.goals)

        # Extract the set of nuts from the goal facts: those with predicate g59iwyxf
        self.goal_nuts = {
            get_parts(goal)[1]
            for goal in self.goal_facts
            if goal.startswith("(g59iwyxf ")
        }

        # Build a map of location connectivity: treat kxokb5hx as undirected edges
        self.location_graph = {}
        for fact in static:
            if fact.startswith("(kxokb5hx "):
                parts = get_parts(fact)
                loc1, loc2 = parts[1], parts[2]
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

        # We will compute shortest paths in the location graph via BFS when needed

        self.shortest_path_cache = {}

    def shortest_distance(self, start, goal):
        """
        Compute the shortest distance between two locations in the location graph using BFS.
        Cache results for efficiency.
        If no path exists, return a large distance.
        """
        if start == goal:
            return 0

        key = (start, goal)
        if key in self.shortest_path_cache:
            return self.shortest_path_cache[key]

        from collections import deque

        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            current, dist = queue.popleft()
            for neighbor in self.location_graph.get(current, ()):
                if neighbor == goal:
                    self.shortest_path_cache[key] = dist + 1
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        # No path found
        self.shortest_path_cache[key] = 1000  # Large number as penalty
        return 1000

    def __call__(self, node):
        state = node.state

        # 1) Check if goal is reached:
        if self.goal_facts <= state:
            return 0

        # Extract nuts that are not yet processed (goal not achieved on them)
        nuts_to_process = [nut for nut in self.goal_nuts if f"(g59iwyxf {nut})" not in state]
        if not nuts_to_process:
            # All nuts processed
            return 0

        # Extract current nuts' locations (u4srx1nb nut location)
        nut_locations = {}
        for fact in state:
            if fact.startswith("(u4srx1nb "):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                if obj in self.goal_nuts:
                    nut_locations[obj] = loc

        # Extract spanner objects and their locations, plus which spanners are in "fresh" state g0brvc06
        spanner_locations = {}
        spanners_with_g0brvc06 = set()
        for fact in state:
            # spanner state
            if fact.startswith("(g0brvc06 "):
                spanner = get_parts(fact)[1]
                spanners_with_g0brvc06.add(spanner)
            # spanner location
            if fact.startswith("(u4srx1nb "):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                if obj in spanners_with_g0brvc06:
                    spanner_locations[obj] = loc

        # The gate location is a location where nuts need to be processed.
        # From the examples and domain file, nuts to be processed are at gate locations.
        # Find gate locations by seeing which locations nuts are at in goals or init.
        # We take the location of the first nut from goal or from nut_locations if present.
        # If not found, fall back to the location of nuts in the initial state.

        # According to domain, nuts are located at 'gate' or some i6azntdy object
        # We treat the gate as given explicitly in instances or initial nut location(s)
        possible_gate_locations = {loc for loc in nut_locations.values()}  # usually one or few gates

        # If no nut locations found, fallback to any nut location in initial_state
        if not possible_gate_locations:
            for fact in state:
                if fact.startswith("(u4srx1nb "):
                    parts = get_parts(fact)
                    obj, loc = parts[1], parts[2]
                    if obj in self.goal_nuts:
                        possible_gate_locations.add(loc)
            if not possible_gate_locations:
                # very unlikely but just to avoid crash
                possible_gate_locations = {"gate"}

        # We'll consider the gate as the location where nuts reside currently (mostly only one)
        gate_locations = possible_gate_locations

        # 2) Estimate spanner moves to gate locations
        # Because spanners must be moved along the location connectivity graph towards the gate to be used.
        # For each spanner, compute minimal distance to one of the gate locations.
        # Then, to approximate minimal total steps, assign spanners to nuts (one spanner per nut)
        # minimizing sum of distances. Use a greedy assignment for efficiency.

        # If no spanners known or no spanner locations, then heuristic must still return a value,
        # so treat as if all spanners are at distance 1000 (large cost).

        if not spanner_locations or not spanners_with_g0brvc06:
            # No usable spanners found, heuristic must still estimate progress:
            # We estimate cost per nut as cost to move a spanner from unknown location (high cost) plus 1 for processing.
            return len(nuts_to_process) * 10

        # Compute distance matrix between spanners and gate locations (min distance to any gate)
        spanner_to_gate_distances = {}
        for spanner, sp_loc in spanner_locations.items():
            dists = [self.shortest_distance(sp_loc, gate) for gate in gate_locations]
            spanner_to_gate_distances[spanner] = min(dists) if dists else 1000

        # Assign spanners greedily to nuts needing processing minimizing spanner travel:
        # Sort spanners by distance ascending
        sorted_spanners = sorted(spanner_to_gate_distances.items(), key=lambda x: x[1])
        spanner_distances_sorted = [dist for _, dist in sorted_spanners]

        # Number of spanners and nuts
        num_spanners = len(spanner_distances_sorted)
        num_nuts = len(nuts_to_process)

        # Sum up the minimal distances for the nuts to be processed:
        # - If nuts ≤ spanners: assign each nut to closest spanner
        # - If nuts > spanners: each spanner may need to process multiple nuts, so multiply distances accordingly
        # This is a heuristic approximation:
        if num_nuts <= num_spanners:
            spanner_cost_for_nuts = sum(spanner_distances_sorted[:num_nuts])
        else:
            # Each spanner handles roughly nut_count / spanner_count nuts
            multiplier = num_nuts / num_spanners
            spanner_cost_for_nuts = sum(dist * multiplier for dist in spanner_distances_sorted)

        # 3) Estimate minimal processing cost per nut
        # Using the domain's 'kjsqfni8' action for nut processing and
        # 'moi6wnfu' and at least one spanner move (n9sbfmrs) per distance unit.
        # We count 1 for the final 'kjsqfni8' action per nut and 1 per 'moi6wnfu' needed.
        # We approximate moving spanners cost included above.

        processing_cost_per_nut = 2  # kjsqfni8 + moi6wnfu minimum

        # 4) Total heuristic is sum of:
        #    - processing costs per nut
        #    - minimal spanner movement costs for all nuts

        heuristic_value = int(processing_cost_per_nut * num_nuts + spanner_cost_for_nuts)

        # Ensure heuristic is at least 1 if goals not reached to guide search
        return max(heuristic_value, 1)
