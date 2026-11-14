from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(go9lczo2 bread1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain.

    Summary
    -------
    This heuristic estimates the number of actions required to fulfill the goal 
    predicate (ivo59apj ?) for all applicable bciid6ad objects (children). 
    The domain's goal basically requires each child to have the (ivo59apj child) 
    fact true.

    Assumptions
    -----------
    - The goal is always to have (ivo59apj ?child) true for a subset of child objects.
    - Each such fact corresponds to completing a chain of actions involving sandwich 
      preparation and delivery to that child.
    - Preparing a sandwich involves three main phases:
      1. "Making" the sandwich: actions kdzjcqwj or h0xkysus that convert initial 
         predicates (go9lczo2 bread, fqungbps content, zogo0zhl sandwich) 
         into predicates i9fq3vit and l2thmd6e (ready/open sandwich).
      2. "Placing" the sandwich on a tray: action jhzk46zy requires a ready sandwich 
         (i9fq3vit) and a tray located on table (y0rxzzmc tray table1). The effect is 
         mpvvhjpt sandwich tray.
      3. "Delivering" the sandwich to the child: actions nfarwtet or xtn2vmez 
         depend on child type predicates (n2ij1fah or dr25ifx3), mpvvhjpt sandwich tray,
         eyker22y child table, and other tray/table predicates, with the effect ivo59apj child.

    Heuristic Initialization
    ------------------------
    - Extract the list of children for which the goal demands (ivo59apj child).
    - Extract initial static predicates for child types n2ij1fah and dr25ifx3.
    - Extract the sets of trays and their current locations (y0rxzzmc) and identify the unique constant bupp7mhz.
    - Extract the relationships eyker22y (child - table).
    - Extract all sandwich objects and their current predicate status from state for heuristic calculations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Calculate how many children still require (ivo59apj child) to be achieved by checking which children from the goal currently do NOT have ivo59apj in state.
    2. For each such child, count the minimal number of actions to deliver a sandwich:
       - To deliver (ivo59apj child), the action nfarwtet or xtn2vmez must be performed, each consuming 1 action.
       - These actions require a sandwich on tray (mpvvhjpt sandwich tray), so action jhzk46zy must be performed if no sandwich is on a tray for delivery.
       - Preparing a sandwich for delivery requires actions kdzjcqwj or h0xkysus, which transform initial bread/content/sandwich states into ready sandwiches.
    3. Current state includes sandwiches that can be ready (i9fq3vit) or currently on trays (mpvvhjpt).
       Use the counts of sandwiches in these states to reduce the number of preparation actions needed.
    4. Assume each remaining child delivery requires roughly 3 actions:
       - 1 for preparation (kdzjcqwj / h0xkysus),
       - 1 for placing sandwich on tray (jhzk46zy),
       - 1 for delivering (nfarwtet/xtn2vmez).
    5. Adjust count by available ready/prepared sandwiches to minimize action count:
       - Use ready sandwiches first,
       - Then sandwiches on trays,
       - Otherwise count full 3 steps per child.
    6. Add small penalties if some prerequisite predicates are missing to increase heuristic guidance.
    7. Return 0 if all children have ivo59apj achieved (goal reached).

    This heuristic is NOT admissible but attempts to balance informativeness and computational efficiency.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - The children that must have ivo59apj in the goal.
        - Static facts specifying child types (n2ij1fah and dr25ifx3).
        - Static facts of eyker22y (child-table relations).
        - The unique tray constant bupp7mhz.
        - Extract and store sets of relevant domain objects and constants.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract children in goal (those that must have ivo59apj)
        self.goal_children = {
            get_parts(goal)[1]
            for goal in self.goals
            if match(goal, "ivo59apj", "*")
        }

        # Extract children with n2ij1fah and dr25ifx3 type predicates (static)
        self.children_n2ij1fah = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "n2ij1fah", "*")
        }
        self.children_dr25ifx3 = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "dr25ifx3", "*")
        }

        # Map child to type for quick access
        self.child_type = {}
        for c in self.children_n2ij1fah:
            self.child_type[c] = "n2ij1fah"
        for c in self.children_dr25ifx3:
            self.child_type[c] = "dr25ifx3"

        # Extract eyker22y relations (child to table)
        self.child_to_table = {}
        for fact in self.static:
            if match(fact, "eyker22y", "*", "*"):
                parts = get_parts(fact)
                child, table = parts[1], parts[2]
                self.child_to_table[child] = table

        # Extract unique zwhbr5cd constant bupp7mhz (tray-related)
        # from domain: (:constants bupp7mhz - zwhbr5cd)
        # this is constant reused in (y0rxzzmc ?tray bupp7mhz)
        self.bupp7mhz = "bupp7mhz"

        # We don't maintain lists of sandwiches/trays here; derived dynamically.

    def __call__(self, node):
        """Compute heuristic value estimating required number of actions to complete goal."""

        state = node.state

        # Quick check if goal reached: all required ivo59apj facts in state
        has_goal = lambda child: f"(ivo59apj {child})" in state
        children_remaining = [c for c in self.goal_children if not has_goal(c)]

        if not children_remaining:
            return 0  # Goal reached exactly

        # We will estimate how many preparation, placing, and delivery actions remain.

        # Count sandwiches ready for placing on tray (i9fq3vit)
        ready_sandwiches = {
            get_parts(fact)[1] for fact in state if match(fact, "i9fq3vit", "*")
        }

        # Count sandwiches on trays (mpvvhjpt ?sandwich ?tray)
        sandwiches_on_trays = {
            get_parts(fact)[1] for fact in state if match(fact, "mpvvhjpt", "*", "*")
        }

        # Count sandwiches that are prepared fully (l2thmd6e) - needed as prerequisite in nfarwtet action
        fully_prepared_sandwiches = {
            get_parts(fact)[1] for fact in state if match(fact, "l2thmd6e", "*")
        }

        # Count sandwiches currently needing preparation (zogo0zhl predicate present)
        sandwiches_to_prepare = {
            get_parts(fact)[1] for fact in state if match(fact, "zogo0zhl", "*")
        }

        # For delivery to each child:
        # If sandwich on tray to child's table, delivery action is possible immediately.
        # Otherwise, sandwich must be prepared + placed on tray + delivered.

        # Track supplies:

        # Number of sandwiches that can be delivered immediately (already on tray)
        # We'll assume any sandwich on any tray can be assigned to one child.
        sandwich_supply_on_trays = len(sandwiches_on_trays)

        # Number of sandwiches ready but not on trays (need placing)
        sandwich_supply_ready = len(ready_sandwiches - sandwiches_on_trays)

        # Number of sandwiches still raw (need preparation)
        sandwich_supply_to_prepare = len(sandwiches_to_prepare)

        # Estimate actions:

        total_actions = 0

        # First consume sandwiches already on tray for delivery:
        deliver_with_sandwich_on_tray = min(len(children_remaining), sandwich_supply_on_trays)
        children_remaining = len(children_remaining) - deliver_with_sandwich_on_tray
        total_actions += deliver_with_sandwich_on_tray  # 1 delivery action per child

        if children_remaining <= 0:
            return total_actions  # All done via delivery actions only

        # Now consume sandwiches that are ready (need placing on tray + delivery)
        place_and_deliver = min(children_remaining, sandwich_supply_ready)
        children_remaining -= place_and_deliver
        # placing = 1 action (jhzk46zy)
        # delivery = 1 action (nfarwtet/xtn2vmez)
        total_actions += place_and_deliver * 2

        if children_remaining <= 0:
            return total_actions

        # Now for remaining children, sandwiches must be prepared, placed on tray, delivered:
        # preparation = 1 action (kdzjcqwj or h0xkysus)
        # placing = 1 action
        # delivery = 1 action
        total_actions += children_remaining * 3

        return total_actions
