from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(y0rxzzmc tray1 bupp7mhz)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False  # not enough parts to match
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (biqhd61z).

    Summary:
    The heuristic estimates the minimum number of actions required to achieve
    the goal of having `(ivo59apj ?child - bciid6ad)` true for all goal children.

    The domain models a process involving sandwiches being prepared from bread and content,
    then trays being assigned, and children eventually receiving the prepared items.
    The main progress is represented by the sequence of predicates:
     - Bread and content available with predicates (go9lczo2 bread) and (fqungbps content)
     - Sandwiches initially raw (zogo0zhl sandwich)
     - Sandwiches get processed (i9fq3vit sandwich)
     - Sandwiches placed on trays (mpvvhjpt sandwich tray)
     - Children in certain states (n2ij1fah or dr25ifx3 child)
     - Children achieve goal predicate (ivo59apj child)

    Assumptions:
    - Each child in the goal must have `(ivo59apj child)` true.
    - Achieving `(ivo59apj child)` requires applying one of `nfarwtet` or `xtn2vmez` actions 
      using corresponding sandwiches and trays.
    - Sandwiches must be prepared first from bread and content using `kdzjcqwj` or `h0xkysus`.
    - Sandwiches need to be "activated" (i9fq3vit) to place them on trays.
    - Trays can be changed dynamically via action `whgyhnl0`.
    - The heuristic does not consider tray replacements, assuming the initial trays suffice.
    - Each sandwich can serve one child.

    Heuristic Initialization:
    - Extract static knowledge of children and their goal status from static facts.
    - Extract goal children from goals.
    - Extract all facts that represent sandwiches and their states.
    - Extract tray objects (from facts about trays associated with bupp7mhz constant).
    - Identify mappings of sandwiches to trays `(mpvvhjpt)`.
    - Identify which children have the goal achieved `(ivo59apj)`.
    - Identify which children are of type `n2ij1fah` or `dr25ifx3` for choosing corresponding goal achievement actions.

    Step-By-Step Thinking for Computing Heuristic:
    1. Count the number of children requiring `(ivo59apj)` goal not yet achieved.
    2. For each such child, one `nfarwtet` or `xtn2vmez` action is needed, requiring:
        a) A sandwich on a tray (mpvvhjpt)
        b) Child matching n2ij1fah (for nfarwtet) or dr25ifx3 (for xtn2vmez).
    3. Count sandwiches currently "activated" (i9fq3vit) to be placed on trays.
       These sandwiches can be turned into serving sandwiches with one `jhzk46zy` action.
    4. To prepare an activated sandwich, it must be prepared using bread and content via
       `kdzjcqwj` or `h0xkysus` actions, which require bread (go9lczo2 predicate) and content (fqungbps)
       and sandwich (zogo0zhl).
    5. Since one sandwich serves one child, the heuristic estimates:
       - Number of children still needing goal served
       - Number of sandwiches currently placed on trays
       - Number of sandwiches activated (but not yet placed)
       - Number of raw sandwiches (zogo0zhl), bread and content pairs available to prepare new sandwiches
    6. The heuristic estimates the number of actions by:
       - Assigning one action for each child to get `(ivo59apj)` (the serving action)
       - One action per sandwich to place it on tray (`jhzk46zy`) if activated
       - Actions to prepare sandwiches from raw ingredients (`kdzjcqwj` or `h0xkysus`), estimated as one action per raw sandwich prepared.
    7. The heuristic assumes a rough linear cost per stage, counting the unmet children and missing sandwiches and needed preparation steps.
    8. Ignore tray replacements (whgyhnl0) in heuristic for efficiency.

    This results in an efficient but possibly non-admissible heuristic that guides the search to:
    - Prepare enough sandwiches,
    - Activate and place them on trays,
    - Serve them to children to complete their goals.
    """

    def __init__(self, task):
        # Store goal children (objects with (ivo59apj child)) required
        self.goal_facts = task.goals

        # Static children of types n2ij1fah and dr25ifx3 for serving actions
        self.n2ij1fah = set()
        self.dr25ifx3 = set()
        for fact in task.static:
            if match(fact, "n2ij1fah", "*"):
                obj = get_parts(fact)[1]
                self.n2ij1fah.add(obj)
            elif match(fact, "dr25ifx3", "*"):
                obj = get_parts(fact)[1]
                self.dr25ifx3.add(obj)

        # Keep the constant for tray identification (bupp7mhz)
        # The heuristic assumes the constant from the domain is unique and fixed.
        self.tray_constant = "bupp7mhz"

    def __call__(self, node):
        state = node.state

        # Helper sets for current state facts:
        # Children that have the goal predicate ivo59apj (already served).
        ivo59apj_children = set()
        for fact in state:
            if match(fact, "ivo59apj", "*"):
                child = get_parts(fact)[1]
                ivo59apj_children.add(child)

        # Children in goal that still need to be served.
        required_goal_children = {
            get_parts(fact)[1]
            for fact in self.goal_facts
            if match(fact, "ivo59apj", "*")
        }
        children_to_serve = required_goal_children - ivo59apj_children

        if not children_to_serve:
            # Goal reached, heuristic is zero.
            return 0

        # Sandwiches currently placed on trays: (mpvvhjpt sandwich tray)
        sandwiches_on_trays = set()
        # Map each sandwich to tray(s) - not necessary for heuristic, so only track sandwiches
        for fact in state:
            if match(fact, "mpvvhjpt", "*", "*"):
                sandwich = get_parts(fact)[1]
                sandwiches_on_trays.add(sandwich)

        # Sandwiches activated but not yet placed on tray: (i9fq3vit sandwich)
        activated_sandwiches = set()
        for fact in state:
            if match(fact, "i9fq3vit", "*"):
                sandwich = get_parts(fact)[1]
                activated_sandwiches.add(sandwich)

        # Raw sandwiches (zogo0zhl sandwich)
        raw_sandwiches = set()
        for fact in state:
            if match(fact, "zogo0zhl", "*"):
                sandwich = get_parts(fact)[1]
                raw_sandwiches.add(sandwich)

        # Bread units with predicates (go9lczo2 bread)
        breads = set()
        for fact in state:
            if match(fact, "go9lczo2", "*"):
                bread = get_parts(fact)[1]
                breads.add(bread)

        # Content units with predicates (fqungbps content)
        contents = set()
        for fact in state:
            if match(fact, "fqungbps", "*"):
                content = get_parts(fact)[1]
                contents.add(content)

        # To prepare a sandwich from raw ingredients using kdzjcqwj or h0xkysus,
        # bread and content must be available and the sandwich raw.
        # Each preparation action consumes one bread and one content, and converts one raw sandwich.

        # Number of sandwiches needed to serve children that are not yet served
        num_children_left = len(children_to_serve)
        num_sandwiches_available = len(sandwiches_on_trays)
        num_sandwiches_activated = len(activated_sandwiches)
        num_sandwiches_raw = len(raw_sandwiches)

        # Actions needed to serve children:
        # Serving children requires one action per child: either nfarwtet or xtn2vmez
        # Depending on child's type (n2ij1fah or dr25ifx3). Count minimal number of children by type.
        children_n2ij1fah = len(children_to_serve & self.n2ij1fah)
        children_dr25ifx3 = len(children_to_serve & self.dr25ifx3)

        # Ensure there are enough sandwiches to serve all children:
        # Each serving action consumes a sandwich on a tray.

        # Sandwiches still needed beyond those currently on trays:
        sandwiches_needed = max(0, num_children_left - num_sandwiches_available)

        # Sandwiches that can be placed on trays via jhzk46zy = activated_sandwiches
        sandwiches_that_can_be_placed = num_sandwiches_activated

        # Sandwiches still needed after placing all activated on trays:
        sandwiches_still_needed_after_placing = max(0, sandwiches_needed - sandwiches_that_can_be_placed)

        # To get new activated sandwiches, raw sandwiches must be prepared using bread and content.
        # Each preparation takes one raw sandwich + one bread + one content and produces one activated sandwich.
        # Because kdzjcqwj or h0xkysus produce i9fq3vit and l2thmd6e, but also remove zogo0zhl, go9lczo2, and fqungbps.

        # Maximum possible prepared activated sandwiches is limited by sandwich raw resources and bread+content pairs.
        max_prepare_possible = min(num_sandwiches_raw, len(breads), len(contents))

        # Number of activated sandwiches currently missing after placing:
        missing_activated_sandwiches = max(0, sandwiches_still_needed_after_placing)

        # Actions count:
        total_actions = 0

        # 1) Serving actions: one per child not served.
        total_actions += num_children_left  # one serving action per child (nfarwtet or xtn2vmez)

        # 2) Placing activated sandwiches on trays: one jhzk46zy action per sandwich placed
        # Place as many activated sandwiches on trays as needed or available.
        place_actions = min(num_sandwiches_activated, sandwiches_needed)
        total_actions += place_actions

        # 3) Preparing raw sandwiches:
        # For each missing activated sandwich, check if preparation possible.
        prepare_actions = min(missing_activated_sandwiches, max_prepare_possible)
        total_actions += prepare_actions  # one prepare action per sandwich

        # Note: If missing_activated_sandwiches > max_prepare_possible,
        # heuristic assumes multiple steps to get resources converted - heuristic caps at finite value.
        # No infinite or zero heuristic for non-goal states.

        # If lacking resources to prepare all needed sandwiches, add heuristic penalty proportional
        # to the gap to encourage getting resources / preparations.

        if missing_activated_sandwiches > max_prepare_possible:
            # Add penalty proportional to missing sandwiches * 2 (arbitrary factor)
            total_actions += 2 * (missing_activated_sandwiches - max_prepare_possible)

        return total_actions
