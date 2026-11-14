from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(ypt9mrgu pjr7u2i2 j0on2kgr)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to achieve all goal facts 
    of form (cb6fw2xd ?loc ?tlfqu) from the current state. The main line of reasoning
    is centered around the "carriage" predicate ypt9mrgu linking k9b7lknr objects to nh1sy8v9 locations,
    the "holding" predicate fguw2142 indicating which location currently "holds" something,
    and the final requirement to have (cb6fw2xd ?loc ?tlfqu) goals satisfied.

    Assumptions
    -----------
    - Each k9b7lknr object appears to represent an entity that must be moved or whose relation
      to nh1sy8v9 objects changes across the plan.
    - The predicate cb6fw2xd (?loc, ?tlfqu) captures the goal condition: these must be true in the goal.
    - Movement and transfer of entities between nh1sy8v9 "locations" are mediated by several actions,
      often requiring the currently held location (fguw2142) to move.
    - Transferring a k9b7lknr from one nh1sy8v9 location to another is a key costly step.
    - Because of the structure of static facts (like s4jrpk2b and ga4l9do0) that define adjacency or connectivity,
      the heuristic integrates them into estimating necessary transfers.
    - The heuristic is not guaranteed to be admissible but aims to provide a close estimate of steps required.

    Heuristic Initialization
    ------------------------
    - Extract goal (cb6fw2xd ?loc ?tlfqu) facts and store them as goals to achieve.
    - Extract static adjacency relations (s4jrpk2b, ga4l9do0, vom4pfig, vjcqgdgn) 
      that describe connectivity between nh1sy8v9 objects.
      These will help estimate minimal moves needed.
    - Index static facts into dictionaries for quick lookup:
        * adjacency sets from s4jrpk2b and ga4l9do0 for estimating reachable locations.
        * Track which tlfqu objects are valid (from fezojtj3).
    - No full path search is done, but adjacency counts guide the heuristic.
    - Extract k9b7lknr objects and qekv4465 objects referenced by goals for focused counting.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Check if current state is goal: if yes, heuristic = 0.
    2. Identify which goal (cb6fw2xd loc tlfqu) facts are missing from the current state.
    3. For each missing goal fact:
       - Find associated k9b7lknr entities (from ypt9mrgu) that might need to be moved 
         to correct nh1sy8v9 location bearing the tlfqu.
       - Estimate the minimal number of transfer actions to place the k9b7lknr at the goal nh1sy8v9.
       - Consider the necessity to move fguw2142 (the "held" status) to intermediate locations
         to allow moves or transfers, roughly estimating 1-2 moves per transfer.
    4. Penalties are increased for missing ypt9mrgu relations aligning with goal goals.
    5. Transfers depend on adjacency predicates (s4jrpk2b, ga4l9do0, vom4pfig, vjcqgdgn),
       which provide notions of connectivity; lack of direct adjacency increases heuristic cost.
    6. Sum estimated transfers and intermediate moves to produce heuristic value.
    7. Return a heuristic estimate that reflects the approximate number of actions to achieve all goal cb6fw2xd facts.

    Note: The heuristic trades off strict admissibility for improved guidance and efficiency.
    """

    def __init__(self, task):
        self.goals = task.goals  # Goal facts (e.g., cb6fw2xd ...)
        self.static = task.static

        # Extract sets of static relations for adjacency and connectivity
        self.s4jrpk2b = set()
        self.ga4l9do0 = set()
        self.vom4pfig = set()
        self.vjcqgdgn = set()
        self.fezojtj3 = set()

        # Also keep track of all static facts by predicate for fast checking
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "s4jrpk2b":
                self.s4jrpk2b.add((parts[1], parts[2]))
            elif pred == "ga4l9do0":
                self.ga4l9do0.add((parts[1], parts[2]))
            elif pred == "vom4pfig":
                self.vom4pfig.add((parts[1], parts[2]))
            elif pred == "vjcqgdgn":
                self.vjcqgdgn.add((parts[1], parts[2]))
            elif pred == "fezojtj3":
                self.fezojtj3.add(parts[1])

        # Index goals of form (cb6fw2xd loc tlfqu)
        self.goal_cb6fw2xd = set()
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "cb6fw2xd":
                self.goal_cb6fw2xd.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state

        # If goal already reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Extract current cb6fw2xd facts, so we know which goal facts remain
        current_cb6fw2xd = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "cb6fw2xd" and len(parts) == 3:
                current_cb6fw2xd.add((parts[1], parts[2]))

        # Missing goal facts of cb6fw2xd
        missing_goals = self.goal_cb6fw2xd - current_cb6fw2xd

        # Extract current ypt9mrgu relations (k9b7lknr linked to nh1sy8v9)
        ypt9mrgu_links = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ypt9mrgu" and len(parts) == 3:
                en18p, stq0j = parts[1], parts[2]
                ypt9mrgu_links.setdefault(en18p, set()).add(stq0j)

        # Extract locations currently "held" (fguw2142 nh1sy8v9)
        held_locations = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "fguw2142":
                held_locations.add(parts[1])

        # We will estimate required actions based on:
        # For each missing (cb6fw2xd loc tlfqu), we try to estimate:
        # - If any k9b7lknr entity needs to be moved to 'loc' to enable the goal.
        # - Number of transfers needed based on adjacency.
        # Each transfer assumed to be minimum 2 actions: 
        # one to move ypt9mrgu from old location to new,
        # plus potential moves of held locations.
        # Also each cb6fw2xd fact requires at least one action (the action that sets it).

        heuristic_value = 0

        # For quick adjacency lookup:
        def adjacent(loc1, loc2):
            return (
                (loc1, loc2) in self.s4jrpk2b
                or (loc1, loc2) in self.ga4l9do0
                or (loc1, loc2) in self.vom4pfig
                or (loc1, loc2) in self.vjcqgdgn
            )

        # For each missing cb6fw2xd goal (location and tlfqu)
        for (goal_loc, goal_tlfqu) in missing_goals:
            # Count one action for achieving cb6fw2xd fact itself
            heuristic_value += 1

            # We want to identify the k9b7lknr entity that should be at goal_loc for this goal_tlfqu.
            # Find all entities en18p that currently have the right tlfqu associated at potentially wrong loc
            # The motions are mostly about moving ypt9mrgu en18p from one loc to goal_loc
            # Check for all en18p that have (qp2t6u4s en18p goal_tlfqu) true (holding that object)
            entities_for_tlfqu = set()
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "qp2t6u4s":
                    en18p, tlfqu = parts[1], parts[2]
                    if tlfqu == goal_tlfqu:
                        entities_for_tlfqu.add(en18p)

            # For each such entity, check its current locations via ypt9mrgu
            # If entity is already at goal_loc, no moves needed for this entity
            # Otherwise estimate minimal transfers needed
            min_transfer_cost = None
            for en18p in entities_for_tlfqu:
                locs = ypt9mrgu_links.get(en18p, set())
                # If already at goal_loc, 0 transfer cost
                if goal_loc in locs:
                    min_transfer_cost = 0 if min_transfer_cost is None else min(min_transfer_cost, 0)
                    continue
                # Otherwise, estimate transfer cost as 2 actions + adjacency penalty
                # For any loc in locs, check adjacency to goal_loc.
                # If adjacent, cost 2; if not, cost 4 (must move through intermediate)
                transfer_cost_for_entity = 4  # default cost for non-adjacent move
                for loc in locs:
                    if adjacent(loc, goal_loc):
                        transfer_cost_for_entity = 2
                        break
                if min_transfer_cost is None or transfer_cost_for_entity < min_transfer_cost:
                    min_transfer_cost = transfer_cost_for_entity

            if min_transfer_cost is None:
                # No entity holds that tlfqu currently, higher penalty for acquiring it
                min_transfer_cost = 6

            heuristic_value += min_transfer_cost

            # Additionally, holding location (fguw2142) sometimes needs to be moved;
            # estimate +1 action if held location not adjacent to goal_loc.
            # Pick one held location only for penalty.
            if held_locations:
                held_loc = next(iter(held_locations))
                if not adjacent(held_loc, goal_loc) and held_loc != goal_loc:
                    heuristic_value += 1

        # To keep heuristic finite, even if no missing goals, we ensure minimum 0
        return max(heuristic_value, 0)
