from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses from a fact string and split into parts."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Return True if the fact's parts match the provided pattern (supports '*' wildcards).
    E.g., match('(at obj loc)', 'at', '*', 'loc') -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (biqhd61z).
    The domain models preparing sandwiches and placing them on trays to serve
    different types of children (two child-types: those that require a
    special sandwich property `l2thmd6e` and those that do not). The
    heuristic estimates how many actions are still required to achieve all
    goal facts of the form (ivo59apj childX) (i.e., to serve every goal child).

    Assumptions
    -----------
    - Each goal is of the form (ivo59apj <child>).
    - Serving a child requires: a sandwich with the appropriate properties
      (l2thmd6e for one child-type), that sandwich placed onto a tray, the
      tray moved to the child's table, and finally the serve action.
    - Resources (bread, content, raw sandwiches, trays) are considered
      optimistically available in the sense that if they are missing we
      add a small penalty to reflect the extra work needed to obtain them
      rather than making the heuristic infinite.
    - We ignore sequencing/resource-contention between multiple goals (we
      compute a simple per-goal minimal-cost estimate and sum them). This
      is not admissible but provides a reasonably informed and cheap-to-
      compute estimate for greedy best-first search.

    Heuristic Initialization
    ------------------------
    The constructor extracts static information from the task:
    - Which children belong to the n2ij1fah (require l2thmd6e) set.
    - Which children belong to the dr25ifx3 set (other child-type).
    - A mapping from child -> table from eyker22y static facts (if present).
    - The goal children (children for which (ivo59apj ...) is required).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goals are already in the state, return 0.
    2. Parse the state to identify:
       - sandwiches that are "ready" (i9fq3vit),
       - sandwiches with the special property (l2thmd6e),
       - sandwiches currently on trays (mpvvhjpt sandwich tray),
       - where trays are located (y0rxzzmc tray table),
       - available raw sandwiches (zogo0zhl),
       - available bread/content facts that affect whether producing a
         special sandwich via kdzjcqwj is possible.
    3. For each goal child not yet served, compute the minimal sequence
       of actions required (optimistically, ignoring interference with
       other goals). We consider (and take the minimum cost among) options:
       - Serve immediately if a valid sandwich is already on a tray at the
         child's table (1 action: serve).
       - Move a tray that already carries a valid sandwich to the child's
         table and serve (2 actions: move + serve).
       - If a valid ready sandwich (i9fq3vit) exists, place it onto a tray
         at bupp7mhz (possibly first moving a tray there), move tray to
         child's table if needed, then serve (2-4 actions depending on
         whether a tray is already at bupp7mhz and whether the tray is
         already at the child's table).
       - Otherwise, produce a sandwich (kdzjcqwj if l2thmd6e required,
         h0xkysus otherwise). Production is assumed to cost 1 action, but
         if required preconditions (raw sandwich, bread, content, or the
         special-content/bread necessary for l2thmd6e) are missing we add
         a small penalty for each missing kind of resource so the heuristic
         remains finite. After production we place/move/serve as above.
    4. Sum the per-child minimal costs to produce the heuristic value.
    This sum is not admissible but provides an informed and efficiently
    computable estimate that favors states closer to having sandwiches
    prepared and already positioned correctly.
    """

    def __init__(self, task):
        """Extract goals and static child/type/table information."""
        self.goals = task.goals  # raw goal fact strings
        self.static = task.static

        # Identify goal children (names only)
        self.goal_children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "ivo59apj" and len(parts) > 1:
                self.goal_children.add(parts[1])

        # Static classification of child types
        self.n2_children = set()  # children with predicate n2ij1fah (need l2thmd6e)
        self.dr_children = set()  # children with predicate dr25ifx3 (other type)
        # static mapping child -> table from eyker22y facts
        self.child_table = {}

        for fact in self.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "n2ij1fah" and len(parts) > 1:
                self.n2_children.add(parts[1])
            elif pred == "dr25ifx3" and len(parts) > 1:
                self.dr_children.add(parts[1])
            elif pred == "eyker22y" and len(parts) > 2:
                # eyker22y child table
                self.child_table[parts[1]] = parts[2]

    def __call__(self, node):
        """Compute the heuristic estimate for the node's state."""
        state = node.state

        # If goals already satisfied, heuristic is zero.
        if all(f"(ivo59apj {child})" in state for child in self.goal_children):
            return 0

        # Parse relevant facts from the current state.
        ready_sandwiches = set()  # i9fq3vit sandw
        special_sandwiches = set()  # l2thmd6e sandw
        raw_sandwiches = set()  # zogo0zhl sandw
        # mpvvhjpt facts: map sandwich -> set of trays carrying it
        sandwich_on_trays = {}
        # tray_location: tray -> table
        tray_location = {}
        # bread/content facts to approximate producibility
        breads_with_go9 = set()  # go9lczo2 bread
        breads_with_e6 = set()  # e6jz04yd bread
        contents_fqu = set()  # fqungbps content
        contents_odd = set()  # odd6wyd1 content

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "i9fq3vit" and len(parts) > 1:
                ready_sandwiches.add(parts[1])
            elif pred == "l2thmd6e" and len(parts) > 1:
                special_sandwiches.add(parts[1])
            elif pred == "zogo0zhl" and len(parts) > 1:
                raw_sandwiches.add(parts[1])
            elif pred == "mpvvhjpt" and len(parts) > 2:
                sand, tray = parts[1], parts[2]
                sandwich_on_trays.setdefault(sand, set()).add(tray)
            elif pred == "y0rxzzmc" and len(parts) > 2:
                tray_location[parts[1]] = parts[2]
            elif pred == "go9lczo2" and len(parts) > 1:
                breads_with_go9.add(parts[1])
            elif pred == "e6jz04yd" and len(parts) > 1:
                breads_with_e6.add(parts[1])
            elif pred == "fqungbps" and len(parts) > 1:
                contents_fqu.add(parts[1])
            elif pred == "odd6wyd1" and len(parts) > 1:
                contents_odd.add(parts[1])

        # Combine static and dynamic child->table mapping (state facts may override/extend)
        # State may also contain eyker22y facts; include them.
        child_table = dict(self.child_table)
        for fact in state:
            if match(fact, "eyker22y", "*", "*"):
                parts = get_parts(fact)
                if len(parts) > 2:
                    child_table[parts[1]] = parts[2]

        # Helper to check if a sandwich is 'valid' for a child (i.e., has l2thmd6e if required)
        def sandwich_valid_for_child(sand, child):
            if child in self.n2_children:
                return sand in special_sandwiches
            # For dr children or unknown type, any sandwich is acceptable
            return True

        # Identify if any tray is at the special constant bupp7mhz (used by jhzk46zy)
        tray_at_bupp = None
        for tray, table in tray_location.items():
            if table == "bupp7mhz":
                tray_at_bupp = tray
                break

        # If no tray exists at all, we still allow moving one with some cost;
        # find any tray that exists in tray_location; otherwise we'll assume
        # a tray can be made available with a small penalty.
        any_tray = next(iter(tray_location.keys()), None)

        total_cost = 0

        # For each goal child, compute minimal cost to serve them from this state.
        for child in self.goal_children:
            # Skip already served children
            if f"(ivo59apj {child})" in state:
                continue

            # Determine child's table (if unknown, assume one move is needed to match)
            child_table_name = child_table.get(child, None)

            # Evaluate options and take minimum
            option_costs = []

            # Option A: There exists a sandwich on some tray that is already suitable.
            for sand, trays in sandwich_on_trays.items():
                if not sandwich_valid_for_child(sand, child):
                    continue
                for tray in trays:
                    tray_tbl = tray_location.get(tray)
                    if tray_tbl is None:
                        # If tray location unknown, penalize but allow
                        # treat as needing 1 move to child's table + serve
                        option_costs.append(2)
                    else:
                        if child_table_name is not None and tray_tbl == child_table_name:
                            # Serve directly
                            option_costs.append(1)
                        else:
                            # Move tray to child's table then serve
                            option_costs.append(2)

            # Option B: There is a ready sandwich (i9fq3vit) suitable for the child (not yet on tray).
            for sand in ready_sandwiches:
                if not sandwich_valid_for_child(sand, child):
                    continue
                # Need: tray at bupp7mhz to use jhzk46zy, otherwise move a tray to bupp7mhz.
                # Costs: (move tray to bupp if needed: maybe 0 or 1) + place (1) + (move to child table if needed: 0 or 1) + serve (1)
                move_to_bupp = 0 if tray_at_bupp is not None else (0 if any_tray is None else 1)
                # If there is no tray at all, assume it can be positioned to bupp7mhz with cost 1.
                if any_tray is None:
                    move_to_bupp = 1
                # After placing, tray will be at bupp7mhz; if child's table is bupp7mhz, no move; else move 1.
                move_to_child = 0 if child_table_name == "bupp7mhz" else (0 if child_table_name is None and any_tray is None else 1)
                # total
                option_costs.append(move_to_bupp + 1 + move_to_child + 1)

            # Option C: Produce a new sandwich (kdzjcqwj if special required, else h0xkysus),
            # then place and serve. We estimate production cost as 1 plus small penalties
            # for missing required resources so the heuristic stays finite.
            needs_special = child in self.n2_children

            # Check producibility for special and non-special sandwich
            # Required categories: raw sandwich, bread (and possibly special bread), content (and possibly special content)
            # We'll count missing categories and add them as penalties.
            missing_penalty = 0

            # raw sandwich present?
            if not raw_sandwiches:
                missing_penalty += 1

            # bread presence
            if needs_special:
                # need a bread that is both go9lczo2 and e6jz04yd
                if not (breads_with_go9 & breads_with_e6):
                    # check if at least go9 or e6 present and penalize accordingly
                    if not breads_with_go9:
                        missing_penalty += 1
                    if not breads_with_e6:
                        missing_penalty += 1
                # content presence: need both fqungbps and odd6wyd1
                if not (contents_fqu & contents_odd):
                    if not contents_fqu:
                        missing_penalty += 1
                    if not contents_odd:
                        missing_penalty += 1
            else:
                # non-special production requires go9lczo2 bread and fqungbps content
                if not breads_with_go9:
                    missing_penalty += 1
                if not contents_fqu:
                    missing_penalty += 1

            # production action cost assumed 1, plus penalties
            production_cost = 1 + missing_penalty

            # After producing, we need to place and serve as in Option B; reuse that cost pattern:
            # compute minimal place/move/serve cost assuming we produced the needed sandwich
            # (same logic as for ready sandwich)
            # Compute move_to_bupp like above
            move_to_bupp = 0 if tray_at_bupp is not None else (0 if any_tray is None else 1)
            if any_tray is None:
                move_to_bupp = 1
            move_to_child = 0 if child_table_name == "bupp7mhz" else (0 if child_table_name is None and any_tray is None else 1)
            place_and_serve_cost = move_to_bupp + 1 + move_to_child + 1
            option_costs.append(production_cost + place_and_serve_cost)

            # As a safety fallback, if no options were found (which shouldn't happen), assign a conservative finite cost.
            if not option_costs:
                option_costs.append(6)

            # Pick minimal option for this child
            child_min_cost = min(option_costs)
            # Accumulate
            total_cost += child_min_cost

        # Ensure non-negative integer
        try:
            return int(total_cost)
        except Exception:
            return 0