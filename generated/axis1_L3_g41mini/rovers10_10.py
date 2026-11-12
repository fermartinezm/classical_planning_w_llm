from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(no_gluten_bread-2 content-portion2 tray3)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain that estimates the
    number of actions needed to achieve the goal.

    Summary
    -------
    The heuristic estimates the minimal number of actions required to transform
    the current state into a goal state by focusing on the outstanding "no_gluten_bread-2"
    facts that must hold on trays. It also considers sandwiches that must be prepared
    and placed on trays according to the goal conditions.

    Assumptions
    -----------
    - The goal requires certain "no_gluten_bread-2" facts for specified content-portions and trays.
    - Making sandwiches requires the availability of content-portions, bread-portions, and sandwiches,
      and the "waiting" and "waiting-2" flags control preparation readiness.
    - Serving sandwiches corresponds to changing certain facts related to gluten allergens.
    - Actions like serving, making sandwiches, moving trays, and putting sandwiches on trays
      each typically correspond to one action count.
    - The heuristic does not need to be admissible but should guide efficient search
      by estimating how many sandwiches/trays still need processing and the complexity of
      gluten-allergic constraints.

    Heuristic Initialization
    ------------------------
    Upon initialization, the heuristic extracts from static facts and goals:
    - The goal trays and content-portions where "no_gluten_bread-2" must hold (i.e., the sandwiches/trays that must be completed).
    - The sets of content-portions, bread-portions, sandwiches, and trays mentioned in static facts and goals.
    - Static "served" relationships between bread-portions.
    - Static "no_gluten_sandwich" facts relating children and bread-portions.
    This information helps determine which bread- and content-portions must be matched and which sandwiches may need to be made.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify from the goal how many "no_gluten_bread-2" (content-portion + tray) facts must be achieved.
       These represent the number of prepared sandwiches on trays we must still achieve.
    2. Count how many "no_gluten_bread-2" facts are already achieved in the current state.
       The difference gives the minimal number of sandwiches/trays still to prepare.
    3. Count how many sandwiches are currently "waiting" (waiting and waiting-2 predicates).
       These usually correspond to sandwiches currently ready or in preparation, influencing how many actions remain.
    4. Consider the presence of "allergic_gluten" and "not_allergic_gluten" facts:
       sandwiches with gluten allergy require extra actions to prepare (e.g., making special sandwiches).
    5. For each tray and sandwich in goal, deduce if sandwiches must be prepared and served,
       counting the minimal number of serve, make, move, and put_on_tray actions.
    6. Approximate the minimum steps left as a function of:
       - Remaining sandwiches/trays to complete (not yet at no_gluten_bread-2).
       - The number of sandwiches waiting or in intermediate preparation steps.
       - The complexity introduced by allergy constraints.
    7. Return this estimated count as the heuristic value.
       The heuristic returns 0 if the goal is already reached.

    This heuristic is not guaranteed admissible but designed to efficiently estimate the remaining
    work to reach the goal with emphasis on the key predicates controlling sandwich preparation and serving.
    """

    def __init__(self, task):
        # Store goal conditions
        self.goals = task.goals
        self.static = task.static

        # Extract goal (content-portion, tray) pairs requiring no_gluten_bread-2
        self.goal_no_gluten_bread_2 = set()
        for fact in self.goals:
            if match(fact, "no_gluten_bread-2", "*", "*"):
                parts = get_parts(fact)
                content_portion = parts[1]
                tray = parts[2]
                self.goal_no_gluten_bread_2.add((content_portion, tray))

        # Extract static facts of served bread-portion pairs (these affect sandwich serving)
        self.served_pairs = set()
        for fact in self.static:
            if match(fact, "served", "*", "*"):
                parts = get_parts(fact)
                self.served_pairs.add((parts[1], parts[2]))

        # Extract static facts regarding gluten allergy status in sandwiches and bread portions
        # Track no_gluten_sandwiches for children & bread-portions, as this relates to special sandwiches
        self.no_gluten_sandwich = set()
        for fact in self.static:
            if match(fact, "no_gluten_sandwich", "*", "*"):
                parts = get_parts(fact)
                child = parts[1]
                bread_portion = parts[2]
                self.no_gluten_sandwich.add((child, bread_portion))

        # Extract all trays appearing in static or goals (for heuristic coverage)
        self.all_trays = set()
        for fact in self.static | self.goals:
            if match(fact, "ontray", "*", "*") or match(fact, "no_gluten_bread-2", "*", "*"):
                parts = get_parts(fact)
                if len(parts) > 1:
                    self.all_trays.add(parts[-1])

    def __call__(self, node):
        state = node.state

        # Quick check if we are at a goal state: heuristic is 0 then
        if self.goals <= state:
            return 0

        # Count no_gluten_bread_2 facts currently achieved (content-portion/tray pairs)
        current_no_gluten_bread_2 = set()
        for fact in state:
            if match(fact, "no_gluten_bread-2", "*", "*"):
                parts = get_parts(fact)
                content_portion = parts[1]
                tray = parts[2]
                current_no_gluten_bread_2.add((content_portion, tray))

        # Remaining no_gluten_bread_2 pairs to satisfy (goal - current)
        remaining_no_gluten_bread_2 = self.goal_no_gluten_bread_2 - current_no_gluten_bread_2
        remaining_trays_to_complete = len(remaining_no_gluten_bread_2)

        # Count sandwiches currently in 'waiting' places (place and content-portion waiting status)
        waiting_places = {get_parts(fact)[1] for fact in state if match(fact, "waiting", "*")}
        waiting_2_pairs = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "waiting-2", "*", "*")}

        # Count how many sandwiches are in preparation or waiting states
        # Approximate: each waiting place probably corresponds to one sandwich being prepared/served
        sandwiches_waiting = len(waiting_places) + len(waiting_2_pairs)

        # Count sandwiches with gluten allergy currently active (allergic_gluten facts)
        allergic_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "allergic_gluten", "*", "*")}

        # Count sandwiches without allergy active (not_allergic_gluten)
        not_allergic = {get_parts(fact)[1] for fact in state if match(fact, "not_allergic_gluten", "*", "*")}

        # Heuristic estimate:
        # This heuristic assumes each remaining no_gluten_bread-2 (tray, content-portion) pair requires
        # at least one make_sandwich or move_tray-2 action, plus serving actions.

        # Rough estimation factors:
        # - One action to make or move tray per remaining sandwich/tray
        # - One action per sandwich waiting or in intermediate step
        # - Extra costs for handling allergic sandwiches (assuming 1 extra action each)
        estimate = 0

        # Base cost: remaining trays to complete
        estimate += remaining_trays_to_complete

        # Plus cost for sandwiches waiting for serving or making
        estimate += sandwiches_waiting

        # Each allergic sandwich possibly requires 1 extra action (more complex preparation)
        estimate += len(allergic_sandwiches)

        # If no trays left but some sandwiches in waiting states, count these as minimum required actions
        if estimate == 0 and (sandwiches_waiting > 0 or len(allergic_sandwiches) > 0):
            estimate = sandwiches_waiting + len(allergic_sandwiches)

        # Ensure heuristic is positive for non-goal non-empty states
        estimate = max(estimate, 1)

        return estimate
