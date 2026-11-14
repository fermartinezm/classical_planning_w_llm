from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate obj1 obj2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (biqhd61z).

    Summary
    -------
    This heuristic estimates the number of actions needed to achieve the goal predicates:
    (ivo59apj <child>). The main goal is to have (ivo59apj ?db93w) for every child specified
    in the goal.

    The domain models a process of preparing sandwiches from bread and content,
    transforming sandwiches, associating them with trays and children, and serving them.
    The key predicates to progress are:
    - Preparing sandwiches (changing zogo0zhl -> i9fq3vit and l2thmd6e)
    - Connecting sandwiches with trays (mpvvhjpt)
    - Serving trays to children (ivo59apj)

    The heuristic estimates how many unresolved children remain to be served plus intermediate steps
    needed for sandwiches in progress.

    Assumptions
    -----------
    - Each "child" (type bciid6ad) must achieve (ivo59apj child).
    - Each (ivo59apj child) requires having a sandwich prepared and placed correctly on a tray associated
      with the child's location.
    - Sandwich preparation actions consume predicates (go9lczo2 bread), (fqungbps content), and (zogo0zhl sandwich).
    - Serving requires trays to be at the right locations (y0rxzzmc tray location).
    - Each child belongs to exactly one location (eyker22y child location).
    - Actions change predicates in fixed ways as per domain definition.
    - Heuristic does not need to be admissible, but efficient and guiding.

    Heuristic Initialization
    ------------------------
    - Extract the list of goal children (i.e., children for which (ivo59apj child) is needed).
    - Extract static mappings for children to locations (eyker22y child location).
    - Extract the known tray location bupp7mhz (constant).
    - Extract the set of all trays known from static y0rxzzmc facts.
    - Extract breads, contents, and sandwiches objects in the static facts (from static and initial states).
    - Precompute sets of predicates indicative of preparation progress for sandwiches. 

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Count how many children still need to be served:
       - For each goal child, check if (ivo59apj child) is true in the current state.
       - Unserved children imply at least one serving action is needed each.

    2. Sandwich preparation progress:
       - Sandwiches start with predicates (zogo0zhl sandwich) unsatisfied after preparation actions.
       - If sandwich has (i9fq3vit sandwich), it is prepared.
       - If sandwich has (l2thmd6e sandwich), it represents a "ready" intermediate state.
       - These predicates indicate partial or full progress towards serving a child.

    3. Trays and sandwich associations:
       - Facts (mpvvhjpt sandwich tray) indicate sandwiches placed on trays.
       - Facts (y0rxzzmc tray location) indicate tray location.
       - Serving requires a prepared sandwich on the tray at the child's location.

    4. Heuristic estimate:
       - Start with count of children still to be served.
       - Add an estimate related to sandwiches:
         * Number of sandwiches not yet prepared (still with zogo0zhl).
         * Sandwiches prepared but not yet placed on a tray (i9fq3vit but not in mpvvhjpt fact).
       - Account for trays not at the child's location (mismatch of tray location to child's location).
       - Combine these counts linearly with weights reflecting minimal actions per step,
         for example:
         - Serving each child: 1 action (final serving)
         - Preparing each sandwich: at least 1-2 actions
         - Placing sandwich on tray: 1 action
         - Moving tray to child's location (if needed): 1 action

    5. Return 0 if all children served (goal state).

    6. The heuristic is purposely non-admissible but efficient to compute.

    Notes
    -----
    - The heuristic focuses on relevant predicates only for efficiency.
    - Neglects complicated action sequences and possible parallelism in the domain.
    """

    def __init__(self, task):
        # Extract all children for which (ivo59apj child) is required in the goal.
        self.goal_children = set()
        for g in task.goals:
            parts = get_parts(g)
            if parts[0] == "ivo59apj" and len(parts) == 2:
                self.goal_children.add(parts[1])

        # Extract static facts so we can know child to location mapping: (eyker22y child location)
        self.child_to_location = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "eyker22y" and len(parts) == 3:
                child, location = parts[1], parts[2]
                self.child_to_location[child] = location

        # Extract trays known from static facts: trays appear with (y0rxzzmc tray location)
        self.known_trays = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "y0rxzzmc" and len(parts) == 3:
                tray = parts[1]
                self.known_trays.add(tray)

        # Constant tray location bupp7mhz (known constant)
        self.constant_location = "bupp7mhz"

        # Precompute sets of predicates relevant to sandwiches and preparation states for quick check
        # We will identify sandwiches by predicates mentioning the sandwich objects' names.

        # We do not know sandwiches upfront, but heuristics only runs from current states.
        # Initialization only sets structures, dynamic extraction happens in __call__.

    def __call__(self, node):
        state = node.state

        # 1. Check which children have been served --> predicate (ivo59apj child)
        served_children = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ivo59apj" and len(parts) == 2:
                served_children.add(parts[1])
        unserved_children = self.goal_children - served_children

        if not unserved_children:
            # Goal reached, heuristic is zero
            return 0

        # 2. Extract sandwich states from predicates:
        # Predicates and their meanings relevant for sandwiches:
        # (zogo0zhl sandwich) => sandwich raw/not prepared
        # (i9fq3vit sandwich) => sandwich prepared
        # (l2thmd6e sandwich) => sandwich ready for serving (intermediate state)
        # (mpvvhjpt sandwich tray) => sandwich placed on tray
        # We gather sandwich objects from predicates referencing sandwiches.

        # Track sandwich preparation status
        sandwiches_raw = set()     # sandwiches with (zogo0zhl)
        sandwiches_prepared = set()  # sandwiches with (i9fq3vit)
        sandwiches_ready_l2thmd6e = set()  # sandwiches with (l2thmd6e)
        sandwiches_on_trays = set()  # sandwiches currently on trays (via mpvvhjpt)

        # For quick lookup of which tray a sandwich is placed on
        sandwich_to_tray = {}

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "zogo0zhl" and len(parts) == 2:
                sandwiches_raw.add(parts[1])
            elif pred == "i9fq3vit" and len(parts) == 2:
                sandwiches_prepared.add(parts[1])
            elif pred == "l2thmd6e" and len(parts) == 2:
                sandwiches_ready_l2thmd6e.add(parts[1])
            elif pred == "mpvvhjpt" and len(parts) == 3:
                sandwich, tray = parts[1], parts[2]
                sandwiches_on_trays.add(sandwich)
                sandwich_to_tray[sandwich] = tray

        # 3. For trays, we check their current location: (y0rxzzmc tray location)
        tray_to_location = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "y0rxzzmc" and len(parts) == 3:
                tray, location = parts[1], parts[2]
                tray_to_location[tray] = location

        # 4. Count sandwiches not prepared (raw) => need preparation actions
        num_raw_sandwiches = len(sandwiches_raw)

        # 5. Count sandwiches prepared but not placed on a tray 
        # We consider sandwiches prepared to be those in sandwiches_prepared or sandwiches_ready_l2thmd6e
        prepared_sandwiches = (sandwiches_prepared | sandwiches_ready_l2thmd6e) - sandwiches_on_trays
        num_prepared_not_on_tray = len(prepared_sandwiches)

        # 6. Count sandwiches on trays but trays possibly not at child's location
        # We assume the sandwich on tray is intended for child's location of the tray
        # Check for sandwiches on trays where tray location != child's location

        # Construct a reverse map of trays associated with children:
        # We can only guess intended children by matching tray location to child's location
        # Because children associated with a location are known (child_to_location),
        # we estimate mismatch cost as number of sandwiches on trays for children not at tray location.

        misplaced_tray_sandwiches = 0
        for sandwich in sandwiches_on_trays:
            tray = sandwich_to_tray[sandwich]
            tray_loc = tray_to_location.get(tray, None)
            # We look which children have tray_loc as their location
            # If no children at tray_loc need serving, count as misplaced

            if tray_loc is None:
                # Missing tray location information, count as misplaced
                misplaced_tray_sandwiches += 1
                continue

            # Check if any unserved child is assigned to that location
            children_at_tray_loc = {c for c, loc in self.child_to_location.items() if loc == tray_loc}
            # Count sandwiches on trays at locations with no children to serve
            # Or if the sandwich does not correspond to any child we still count it as misplaced

            # We approximate: if no unserved child at tray_loc, tray is misplaced
            if not (children_at_tray_loc & unserved_children):
                misplaced_tray_sandwiches += 1

        # 7. Estimate heuristic cost:
        # Weighting:
        # - Each unserved child likely requires at least 1 action (serving)
        # - Each raw sandwich requires at least 2 actions (prepare sandwich with kdzjcqwj or h0xkysus)
        # - Each prepared but not placed sandwich requires at least 1 action (place on tray)
        # - Each misplaced tray sandwich requires at least 1 action (move tray to right location)

        est = (
            len(unserved_children) * 1 +
            num_raw_sandwiches * 2 +
            num_prepared_not_on_tray * 1 +
            misplaced_tray_sandwiches * 1
        )

        # Minimum heuristic value is 1 if not zero (goal), to ensure non-zero estimates for progress states.
        return max(est, 1)
