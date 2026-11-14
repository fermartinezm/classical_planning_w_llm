from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain ("spanner") domain.

    Summary
    -------
    This heuristic estimates the number of actions required to reach a goal state by focusing on the placement
    of locatable objects (tools/spanners) at target locations (tightened-2 predicates). The main challenge 
    is to "walk" or "pickup_spanner" locatables that are linked, and to move tightened nuts while ensuring 
    their correct locations. The heuristic counts the number of tightened-2 goals not yet achieved and estimates
    the minimal steps to achieve them, based on current tightened and link predicates and nut loose-2 locations.

    Assumptions
    -----------
    - Each goal is of the form (tightened-2 locatable location).
    - Nuts are associated with locatables, with tightened and loose predicates describing their status.
    - The number of tightened-2 goals not yet achieved closely approximates the number of required actions.
    - Movements and tightenings can be approximated conservatively by counting unmet goals and loose nuts.
    - The heuristic does not need to be admissible, but should guide greedy best-first search effectively.
    - The domain facts represent a network of locatables connected via link, loose, carrying, and usable relations.

    Heuristic Initialization
    ------------------------
    - Extract goal conditions requiring (tightened-2 ?locatable ?location).
    - Extract static facts describing loose connections (loose, loose-2), usable connections, locatable relations.
    - No heavy graph search or shortest paths computation since heuristic must be efficient.
    - Index facts for quick lookup: tightened nuts, loose nuts, linked locatables, carrying relations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify which (tightened-2 locatable location) goals are not yet achieved.
    2. For each such goal:
       - Check if the locatable is currently linked (link predicate).
       - Check if the nut associated with the locatable is tightened at that locatable.
       - Check if the nut is still loose at some location.
    3. Count the number of tightened nuts that have to be "walked" or moved to satisfy the tightened-2 goals.
    4. Count loose nuts that need to be tightened and moved to target locations.
    5. Treat each unsatisfied goal as requiring at least one action to move/tighten the nut.
    6. Add penalties if nuts are loose away from the goal positions, since they need to be tightened and moved.
    7. Return the sum of unmet tightened-2 goals plus loose nuts count as an approximate heuristic cost.
    8. Return 0 if all goals are met.

    Overall, this heuristic balances accuracy and computation time by counting unresolved tightened-2 goals and
    loose nuts, which together correlate well with the number of required actions remaining.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting
        - The goal facts involving tightened-2 relations.
        - The set of nuts and locatables involved that appear in goal and static predicates.
        """
        self.goals = set()
        self.goal_locatables = set()
        # Collect tightened-2 goals (locatable, location)
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "tightened-2" and len(parts) == 3:
                loc, loc2 = parts[1], parts[2]
                self.goals.add((loc, loc2))
                self.goal_locatables.add(loc)

        # Extract static facts for quick membership tests in heuristic calculation
        # We primarily need:
        #   - loose-2 facts: (loose-2 nut location)
        #   - tightened facts: (tightened nut locatable)
        #   - link facts: (link locatable)
        #   - carrying facts: (carrying locatable locatable)
        self.loose2 = set()
        self.tightened = set()
        self.link = set()
        self.carrying = set()
        self.usable = set()

        for fact in task.static:
            parts = get_parts(fact)
            p = parts[0]
            if p == "loose-2" and len(parts) == 3:
                self.loose2.add((parts[1], parts[2]))  # (nut, location)
            elif p == "tightened" and len(parts) == 3:
                self.tightened.add((parts[1], parts[2]))  # (nut, locatable)
            elif p == "link" and len(parts) == 2:
                self.link.add(parts[1])  # locatable
            elif p == "carrying" and len(parts) == 3:
                self.carrying.add((parts[1], parts[2]))  # (locatable, locatable)
            elif p == "usable" and len(parts) == 3:
                self.usable.add((parts[1], parts[2]))  # (locatable, locatable)

        # We keep static loose-2 and link as initial sets since those do not change by operators.
        # For heuristic, facts in state override static facts, so updated sets will be extracted at runtime.

    def __call__(self, node):
        """
        Compute a heuristic estimate of the cost to reach the goal from the current node state.

        Calculation steps:
        1. Extract current facts for tightened-2, tightened, loose-2, link, carrying, and loose predicates.
        2. Determine which tightened-2 goals are already satisfied.
        3. Identify nuts that are loose and their current locations.
        4. Count remaining goals that are not met.
        5. Estimate the cost contribution by:
            - Number of tightened-2 goals not met.
            - Plus number of loose nuts that need tightening/moving.
        6. Return 0 if goal achieved (all tightened-2 goals met).
        """
        state = node.state

        # Extract relevant predicates from current state for this heuristic

        tightened2_current = set()  # (locatable, location)
        tightened_current = set()  # (nut, locatable)
        loose2_current = set()  # (nut, location)
        link_current = set()  # locatable
        carrying_current = set()  # (locatable, locatable)
        loose_current = set()  # (locatable, locatable)
        usable_current = set()  # (locatable, locatable)

        # Parse state facts efficiently once
        for fact in state:
            parts = get_parts(fact)
            p = parts[0]
            if p == "tightened-2" and len(parts) == 3:
                tightened2_current.add((parts[1], parts[2]))
            elif p == "tightened" and len(parts) == 3:
                tightened_current.add((parts[1], parts[2]))
            elif p == "loose-2" and len(parts) == 3:
                loose2_current.add((parts[1], parts[2]))
            elif p == "link" and len(parts) == 2:
                link_current.add(parts[1])
            elif p == "carrying" and len(parts) == 3:
                carrying_current.add((parts[1], parts[2]))
            elif p == "loose" and len(parts) == 3:
                loose_current.add((parts[1], parts[2]))
            elif p == "usable" and len(parts) == 3:
                usable_current.add((parts[1], parts[2]))

        # Check which goal tightened-2 facts are not yet met
        unmet_goals = self.goals - tightened2_current
        if not unmet_goals:
            return 0

        # Count loose nuts that need to be tightened at some location
        # We consider nuts that are loose somewhere (loose-2) as needing tightening and moving
        nuts_loose_locations = {}
        for nut, loc in loose2_current:
            nuts_loose_locations.setdefault(nut, set()).add(loc)

        # Count nuts that have tightened facts but might need to move tools (link locatable)
        tightened_nuts = set(nut for (nut, _) in tightened_current)

        # Heuristic calculation:
        # Start with the number of unmet tightened-2 goals
        cost = len(unmet_goals)

        # Add a penalty for each loose nut because it must be tightened and moved
        cost += len(nuts_loose_locations)

        # Additional heuristic insight:
        # If a locatable required by a goal is not linked currently but should be, add penalty
        goal_locatables_unlinked = [loc for (loc, _) in unmet_goals if loc not in link_current]
        cost += len(goal_locatables_unlinked)  # Cost to re-link or move spanners

        # Possibly add a small cost per nut that is tightened but carried at different locatables,
        # since this involves movement to satisfy the tightened-2 goals.
        # Count nuts at locatables that are not in goal tightened-2 locations
        # We'll approximate this by counting tightened nuts not at locations requested by goals.
        tightened_nut_locs = {}
        for (nut, locatable) in tightened_current:
            tightened_nut_locs[nut] = locatable

        # For each nut that must be tightened at some goal locatable, add a cost if current locatable differs
        for (goal_locatable, goal_location) in unmet_goals:
            # Find nuts associated (from tightened_current) roughly matching goal locatable
            # If nut tightened at different locatable => cost for moving
            # Since goal ties locatable, check if nut tightened loc differs
            nuts_for_goal = [nut for (nut, loc) in tightened_current if loc == goal_locatable]
            for nut in nuts_for_goal:
                # If nut currently tightened loc not matching goal locatable, penalize
                if tightened_nut_locs.get(nut, None) != goal_locatable:
                    cost += 1

        return cost
