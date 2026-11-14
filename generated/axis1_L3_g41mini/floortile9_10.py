from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (spanner) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve all goal
    facts of the form `(tightened-2 locatable location)` from the current state.
    The core idea focuses on how many locatable objects must be "moved" and/or "tightened"
    in the correct location, considering the linking and carrying structure between locatables,
    and the loose vs tightened state of nuts. Since the goal is about `tightened-2` bindings,
    the heuristic tries to estimate the minimal number of moves and tightening actions needed
    to move each relevant locatable from its current state to its goal location.

    Assumptions
    -----------
    - Goal facts are only `(tightened-2 locatable location)` predicates.
    - Each locatable in the goal must be tightened in the specified location.
    - The domain's main complexity lies in "walking"/"carrying" and "tightening" nuts and locatables
      through links defined by the `link`, `carrying` and `loose` predicates.
    - Moving and tightening actions roughly correspond to moving `link` or tightening a nut
      to a locatable in the right location.
    - The heuristic does not guarantee admissibility but tries to minimize expanded nodes by
      estimating minimal sequences of tightening and walking actions for each goal locatable.

    Heuristic Initialization
    -------------------------
    From the static facts, the heuristic extracts:
    1. `loose` relations between locatables (used to understand adjacency).
    2. `at` relations among locatables to build a connectivity graph.
    3. `carrying` relations that specify which locatables can carry others.
    4. `usable` relations that indicate edges usable for tightening_nut-3 actions.
    5. All known locatable objects.
    6. The fixed `at-2` locations to identify current locations of nuts.

    It also processes the goal facts to extract a mapping from locatables to their desired tightened locations.

    Step-By-Step Thinking for Computing Heuristic
    ----------------------------------------------
    1. Identify for each goal `(tightened-2 locatable goal_location)` whether the locatable
       is already in the correct `tightened-2` relation (cost 0 for that locatable).
    2. For locatables not yet in the correct tightened-2 state:
       a) Check if the locatable is currently linked (`link`) or carrying another locatable,
          and if it can be "walked" or "carried" to the goal location.
       b) Use the static `loose` and `at` relations to build a graph indicating reachable locatables,
          modelling the `walk` and `pickup_spanner` actions.
       c) Estimate the minimal number of moves to get the locatable linked in the goal location,
          counting at least one action per move along the loose links.
    3. Count the nuts that are loose (`loose-2`) but must be tightened to a location:
       - Each loose nut requires at least one `tighten_nut` action.
    4. For each locatable, estimate additionally the minimum required tightening actions (`tighten_nut-2` or `tighten_nut-3`)
       to match the goal state. This is at least one action per locatable if not already tightened.
    5. Sum these minimal estimated actions over all locatables/nuts involved in the goal, yielding the heuristic value.

    Notes:
    - The heuristic is 0 if and only if all goal facts are true (goal reached).
    - The heuristic never returns infinity; reduces to counting minimal moves and tighten actions.
    - It leverages the static graph structure from `loose`, `at`, `carrying`, and `usable` facts for connectivity.
    """

    def __init__(self, task):
        # Extract goals of the form (tightened-2 locatable location)
        self.goals = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "tightened-2" and len(parts) == 3:
                locatable, location = parts[1], parts[2]
                self.goals[locatable] = location

        # Extract static info
        static_facts = task.static

        # Build adjacency graph of locatables based on "loose" relations (bidirectional)
        self.loose_adj = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "loose" and len(parts) == 3:
                a, b = parts[1], parts[2]
                self.loose_adj.setdefault(a, set()).add(b)
                self.loose_adj.setdefault(b, set()).add(a)

        # Map each locatable's containing locatable(s) via "at" edges
        self.at_adj = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                inner, outer = parts[1], parts[2]
                self.at_adj.setdefault(inner, set()).add(outer)

        # Map carrying relationships between locatables
        self.carrying = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "carrying" and len(parts) == 3:
                carrier, carried = parts[1], parts[2]
                self.carrying[carried] = carrier

        # Store all usable edges (for tightening nut-3)
        self.usable = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "usable" and len(parts) == 3:
                loc, loc2 = parts[1], parts[2]
                self.usable.setdefault(loc, set()).add(loc2)

    def __call__(self, node):
        state = node.state

        # Quick check: if all goal facts are in the state, heuristic is zero
        if all(f"(tightened-2 {loc} {loc_loc})" in state for loc, loc_loc in self.goals.items()):
            return 0

        # Extract current tightened-2 relations in state
        tightened_2_current = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "tightened-2" and len(parts) == 3:
                tightened_2_current[parts[1]] = parts[2]

        # Extract tightened facts (tightened nut locatable)
        tightened_current = set(
            fact for fact in state if fact.startswith("(tightened ")
        )

        # Extract loose-2 nuts and their location
        loose_2 = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "loose-2" and len(parts) == 3:
                loose_2[parts[1]] = parts[2]

        # Extract all locatables that are currently linked
        linked_locatables = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "link" and len(parts) == 2:
                linked_locatables.add(parts[1])

        # Helper function: Estimate shortest path length on loose_adj graph (undirected),
        # between two locatables (or None if no path).
        # We treat the "loose" relation as adjacency (step cost 1 per move).
        def shortest_path_length(start, goal):
            if start == goal:
                return 0
            visited = {start}
            frontier = [(start, 0)]
            while frontier:
                current, dist = frontier.pop(0)
                neighbors = self.loose_adj.get(current, set())
                for n in neighbors:
                    if n == goal:
                        return dist + 1
                    if n not in visited:
                        visited.add(n)
                        frontier.append((n, dist + 1))
            return None  # No path found

        '''
        Heuristic strategy:
        - For each goal locatable and its goal location (tightened-2 locatable location):
          * If locatable is already in position tightened-2 locatable location, cost for it = 0.
          * Otherwise:
              - Estimate minimal moves to get locatable linked in goal location. A move is walking or carrying.
                We approximate this by shortest path length between where locatable currently is linked and goal location.
                If locatable is not linked, consider it needs to become linked first (extra cost).
              - Count at least one action to tighten it in location if not tightened-2.
        - For each nut that is loose-2 (loose) but needs to be tightened (i.e. loose-2 nut location != at-2 location or goal location),
          add cost to tighten nut at least once.
        '''

        total_cost = 0

        # Map current tightened locatable to location via tightened-2 (if any)
        # Also get a heuristic guess of current locatable's location for path computations:
        # If tightened-2 present, use its location; else fallback to linked locatable or loose-2 location.
        def get_current_locatable_location(loc):
            # Current location by tightened-2 predicate
            loc_loc = tightened_2_current.get(loc)
            if loc_loc:
                return loc_loc
            # If loc is linked, cannot infer exact location from link, use loc itself as proxy
            if loc in linked_locatables:
                return loc  # best guess: itself
            # If loc is carried by another locatable, recursively find location of carrier
            carrier = self.carrying.get(loc)
            if carrier:
                return get_current_locatable_location(carrier)
            # If still no info, fallback to loc itself (conservative)
            return loc

        for loc, goal_loc in self.goals.items():
            current_loc = get_current_locatable_location(loc)
            # Estimate moves cost: shortest path on loose graph from current_loc to goal_loc
            path_len = shortest_path_length(current_loc, goal_loc)

            # If no path found, assume at least 2 actions: one to link/prepare, one to move or tighten
            if path_len is None:
                path_len = 2

            # If locatable is already tightened-2 at goal location, no penalty, else count 1 tightening action
            tightened_2_at_goal = (
                tightened_2_current.get(loc) == goal_loc
            )
            tighten_cost = 0 if tightened_2_at_goal else 1

            # Sum minimum estimated movements plus one tightening per locatable needing goal
            total_cost += path_len + tighten_cost

        # For each loose-2 nut that is not yet tightened (i.e., no (tightened nut locatable) fact), add at least one action
        # to tighten nut in some location (needed before it can be used)
        nuts_in_goal = set()
        for fact in self.goals:
            if fact.startswith("nut"):
                nuts_in_goal.add(fact)
        # But goal facts are only tightened-2 locatable location so nuts may not appear there;
        # instead, consider all nuts in loose-2 and check if they are tightened at some locatable.

        nuts_to_tighten = 0
        # Find all nuts mentioned loosely (loose-2) and tightened
        nuts_loose = set(loose_2.keys())
        nuts_tightened = set()
        for fact in tightened_current:
            parts = get_parts(fact)
            if parts[0] == "tightened" and len(parts) == 3:
                nuts_tightened.add(parts[1])

        for nut in nuts_loose:
            if nut not in nuts_tightened:
                # Nut is loose; requires at least one action to tighten it
                nuts_to_tighten += 1

        total_cost += nuts_to_tighten

        # The heuristic must be finite and >= 0
        if total_cost < 0:
            total_cost = 0

        return total_cost
