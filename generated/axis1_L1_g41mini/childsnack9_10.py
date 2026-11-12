from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(go9lczo2 bread1)".
    - `args`: The expected pattern, e.g. ("go9lczo2", "*").
    - Returns True if the fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (biqhd61z).

    Summary
    -------
    This heuristic estimates how many actions are needed to reach the goal state
    where certain children have the fact (ivo59apj ?child) set. The domain models
    a preparation and delivery process involving sandwiches, bread, contents, trays,
    and children. The goal is achieved when each specified child has the fact (ivo59apj).

    Assumptions
    -----------
    - Each child must have (ivo59apj child) true at the goal.
    - To achieve (ivo59apj child), a sandwich must have been prepared (i9fq3vit and l2thmd6e facts on sandwiches),
      placed on a tray (mpvvhjpt), and the child-table and tray-table relations (eyker22y and y0rxzzmc) must be consistent.
    - Sandwich preparation consumes bread and content facts and changes (zogo0zhl) and others.
    - There are two types of children (from static facts): those with (n2ij1fah) and those with (dr25ifx3), 
      which correspond to two different delivery actions with slightly different preconditions.
    - Bread and content can only participate in actions if certain facts hold (like go9lczo2, fqungbps, odd6wyd1, e6jz04yd).
    - Trays can change the associated location fact (y0rxzzmc).

    Heuristic Initialization
    ------------------------
    - Extract and store the set of children required to have (ivo59apj) in the goal.
    - Use static facts to distinguish children of type n2ij1fah and dr25ifx3.
    - Extract the (eyker22y child tray) relations for children to known locations.
    - Extract which trays are associated with bupp7mhz (the constant location).
    - Precompute the sets of bread facts with e6jz04yd and those with go9lczo2 for quick checks.
    - Extract content facts with odd6wyd1 and fqungbps.
    - Sandwiches currently prepared (i9fq3vit), and sandwiches ready (l2thmd6e) or not (zogo0zhl).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. For each child that still lacks (ivo59apj), estimate how many steps are needed to serve it.
       Serving a child involves:
         a) Having a prepared sandwich that meets certain conditions
         b) Associating the sandwich with a tray at the child's location
         c) Delivering the sandwich to the child
    2. From the state, identify how many sandwiches are currently prepared and ready to be delivered.
       (Prepared means (i9fq3vit) true; ready means also (l2thmd6e) true, which is needed for n2ij1fah children)
    3. Count how many sandwiches remain to be prepared (those sandwiches can only be made if bread with go9lczo2 and e6jz04yd,
       contents with fqungbps and odd6wyd1, and sandwiches with zogo0zhl are available).
    4. The main actions contributing to the heuristic count are:
         - Preparing sandwiches (actions kdzjcqwj or h0xkysus)
         - Placing sandwiches on trays (jhzk46zy)
         - Delivering sandwiches to each child (nfarwtet or xtn2vmez)
         - Possibly action whgyhnl0 that changes tray locations but assumed less frequent or can be ignored for heuristic
    5. Estimate minimum required actions for leftover children as:
         - One delivery action per undelivered child (either nfarwtet or xtn2vmez depending on child type)
         - One action placing sandwich on tray per sandwich to deliver
         - One action preparing sandwich per sandwich needed
    6. Sandwiches prepared but not ready (missing l2thmd6e) must be prepared with kdzjcqwj to gain l2thmd6e.
    7. The heuristic sums:
         actions_to_prepare_sandwiches + actions_to_place_on_trays + delivery_actions_per_missing_child
    8. The heuristic is 0 if all children have the goal fact (ivo59apj).

    This heuristic is not guaranteed admissible but aims to be informative and efficiently computed
    to reduce search expansions.
    """

    def __init__(self, task):
        # Goals: children that must have (ivo59apj ?child)
        self.goal_children = set()
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "ivo59apj":
                self.goal_children.add(parts[1])

        self.static = task.static

        # Identify static info sets
        # Children types
        self.children_n2ij1fah = set()
        self.children_dr25ifx3 = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "n2ij1fah":
                self.children_n2ij1fah.add(parts[1])
            elif parts[0] == "dr25ifx3":
                self.children_dr25ifx3.add(parts[1])

        # eyker22y(child, tray_location)
        self.child_table = dict()  # child -> tray location (zwhbr5cd)
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "eyker22y":
                self.child_table[parts[1]] = parts[2]

        # Note: bupp7mhz constant of type zwhbr5cd (location for trays)
        # No static facts about trays location except fact y0rxzzmc(tray, bupp7mhz) from initial state

    def __call__(self, node):
        state = node.state

        # Sets for quick membership tests
        facts = set(state)

        # Which children already have the goal fact (ivo59apj)
        served_children = {fact for fact in facts if fact.startswith("(ivo59apj ")}
        served_children_set = {get_parts(fact)[1] for fact in served_children}

        # Children yet to serve (from goal but not served yet)
        remaining_children = self.goal_children - served_children_set
        if not remaining_children:
            # Goal reached: heuristic zero.
            return 0

        # Extract sandwiches:
        # i9fq3vit ?yam4s - sandwich prepared
        prepared_sandwiches = {get_parts(fact)[1] for fact in facts if match(fact, "i9fq3vit", "*")}
        # l2thmd6e ?yam4s - sandwich ready (only those with this are ready to be used for children n2ij1fah)
        ready_sandwiches = {get_parts(fact)[1] for fact in facts if match(fact, "l2thmd6e", "*")}
        # zogo0zhl ?yam4s - sandwich not prepared
        unprepared_sandwiches = {get_parts(fact)[1] for fact in facts if match(fact, "zogo0zhl", "*")}

        # Trays currently on bupp7mhz (location)
        trays_on_location = {get_parts(fact)[1] for fact in facts if match(fact, "y0rxzzmc", "*", "bupp7mhz")}

        # Sandwiches placed on trays (mpvvhjpt ?yam4s ?zw4bi)
        sandwiches_on_trays = {}
        for fact in facts:
            if match(fact, "mpvvhjpt", "*", "*"):
                parts = get_parts(fact)
                sandwiches_on_trays[parts[1]] = parts[2]  # sandwich -> tray

        # Bread facts in state:
        breads_with_go9lczo2 = {get_parts(fact)[1] for fact in facts if match(fact, "go9lczo2", "*")}
        breads_with_e6jz04yd = {get_parts(fact)[1] for fact in facts if match(fact, "e6jz04yd", "*")}
        # Contents facts in state:
        contents_with_fqungbps = {get_parts(fact)[1] for fact in facts if match(fact, "fqungbps", "*")}
        contents_with_odd6wyd1 = {get_parts(fact)[1] for fact in facts if match(fact, "odd6wyd1", "*")}

        # Count sandwiches that can be prepared currently:
        # Condition: bread with go9lczo2 and e6jz04yd, content with fqungbps and odd6wyd1, sandwich unprepared (zogo0zhl)
        # Since parameters are abstracted, approximate by counting minimum among available
        num_bread_ready = min(len(breads_with_go9lczo2 & breads_with_e6jz04yd), len(unprepared_sandwiches))
        num_content_ready = min(len(contents_with_fqungbps & contents_with_odd6wyd1), len(unprepared_sandwiches))
        possible_preparations = min(num_bread_ready, num_content_ready)

        # Count how many sandwiches remain unprepared that could be prepared
        sandwiches_unprepared_and_possible = min(possible_preparations, len(unprepared_sandwiches))

        # Count sandwiches prepared but not ready (missing l2thmd6e)
        sandwiches_prepared_not_ready = prepared_sandwiches - ready_sandwiches

        # Count sandwiches ready but not placed on trays:
        sandwiches_ready_and_prepared = ready_sandwiches & prepared_sandwiches
        sandwiches_ready_unplaced = sandwiches_ready_and_prepared - set(sandwiches_on_trays.keys())

        # Children delivery count:
        # We assume minimum one delivery action per remaining child
        num_children_remaining = len(remaining_children)

        # Actions estimation:

        # 1) Actions to prepare sandwiches:
        # Each sandwich preparation (kdzjcqwj or h0xkysus) counts as 1 action.
        # We count how many sandwiches need preparation to supply the remaining children.
        # Number required at least = remaining children - sandwiches_ready_and_prepared that can be used (including those placed on trays)
        # But sandwiches placed on trays can only serve one child each.

        # Count how many sandwiches can be used immediately (ready and placed on trays)
        sandwiches_placed = set(sandwiches_on_trays.keys())
        usable_sandwiches = sandwiches_ready_and_prepared & sandwiches_placed

        # Number of children that can be served immediately: count of usable sandwiches
        children_served_by_ready_sandwiches = len(usable_sandwiches)

        # Children remaining to serve after using ready sandwiches
        children_to_serve_after_ready = max(num_children_remaining - children_served_by_ready_sandwiches, 0)

        # Need to prepare or ready sandwiches for these children
        # We can use sandwiches that are prepared but not ready by applying kdzjcqwj (1 action per sandwich)
        # Or prepare unprepared sandwiches (1 action each, but needs content and bread, so limited by possible_preparations)

        # Use prepared but not ready sandwiches first:
        num_prep_not_ready_used = min(len(sandwiches_prepared_not_ready), children_to_serve_after_ready)
        children_to_serve_after_ready -= num_prep_not_ready_used

        # For remaining children, prepare new sandwiches:
        num_new_preparations = min(sandwiches_unprepared_and_possible, children_to_serve_after_ready)

        # Total preparations action cost:
        actions_prepare = num_prep_not_ready_used + num_new_preparations

        # 2) Actions to place sandwiches on trays:
        # Each sandwich placed on tray requires one action (jhzk46zy).
        # Estimate number of sandwiches to place on trays for children not yet served by placed sandwiches.
        # These children are equal to remaining_children - children_served_by_ready_sandwiches

        sandwiches_currently_placed = len(sandwiches_on_trays)
        sandwiches_needed_placed = children_to_serve_after_ready + num_prep_not_ready_used  # sandwiches we will use now need to be placed

        # Since some sandwiches ready but unplaced exist, they need placing
        sandwiches_ready_unplaced_count = len(sandwiches_ready_unplaced)

        # We take max between sandwiches_needed_placed and sandwiches_ready_unplaced_count
        # But not fewer than zero
        actions_place_on_trays = max(sandwiches_ready_unplaced_count, sandwiches_needed_placed)
        # If no sandwiches need to be placed, cost is zero.

        # 3) Delivery actions:
        # One delivery action (nfarwtet or xtn2vmez) per remaining child
        actions_delivery = num_children_remaining

        # Optional:
        # We do not estimate whgyhnl0 (tray location switch) because we expect minimal effect on heuristic

        # Total heuristic estimation:
        heuristic_value = actions_prepare + actions_place_on_trays + actions_delivery

        # Return heuristic_value as an integer >= 0
        return heuristic_value
