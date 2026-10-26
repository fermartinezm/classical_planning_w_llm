from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    who are currently waiting and not yet served. It considers the steps needed
    to prepare suitable sandwiches (gluten-free if required), place them on
    trays, move trays to the correct tables, and finally serve the children.
    The heuristic sums up the estimated costs for making sandwiches, putting
    them on trays, moving trays, and serving children, accounting for
    sandwich types (gluten-free vs. regular) and the current state of
    sandwiches and trays.

    # Assumptions
    - The primary goal is to serve all children specified in the task goals.
    - Children remain waiting at their initial tables until served.
    - Trays, bread, and content are initially available in the kitchen.
    - Enough components exist in solvable problems to make all required sandwiches.
    - Trays moved to tables remain at tables.
    - Sandwiches are consumed upon serving a child.
    - The costs for actions are:
        - get_bread_kitchen: 1
        - get_content_kitchen: 1
        - make_sandwich: 1
        - put_on_tray: 1
        - move_tray_kitchen_to_table: 1
        - serve_child: 1
    - Making a sandwich requires 3 actions (get bread, get content, make).

    # Heuristic Initialization
    The heuristic extracts the following information from the task definition:
    - The set of children who need to be served (from task goals).
    - The allergy status (allergic_gluten) for each child (from static facts).
    - The table where each child is waiting (from static facts).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:

    1.  Identify the set of children who are goals but are not yet marked as `served` in the current state. Let this set be `U`.
    2.  If `U` is empty, the heuristic is 0 (goal state).
    3.  Initialize the total heuristic cost `H` to 0.
    4.  Add the cost for the final `serve_child` action for each unserved child. `H += len(U)`.
    5.  Identify the tables where children in `U` are waiting (using static `waiting` facts). Let this set be `Tables_with_waiting_U`.
    6.  Identify the tables where trays are currently located (`at_table` predicate in the state). Let this set be `Tables_with_trays`.
    7.  Count the number of tables in `Tables_with_waiting_U` that are not in `Tables_with_trays`. These tables will require a tray to be moved from the kitchen. Add the cost for these tray movements. `H += len(Tables_with_waiting_U - Tables_with_trays)`.
    8.  Separate the unserved children `U` into those needing gluten-free sandwiches (`U_gf`) and those needing regular sandwiches (`U_reg`), based on static `allergic_gluten` facts.
    9.  Count the total number of GF sandwiches needed (`needed_gf = len(U_gf)`) and Regular sandwiches needed (`needed_reg = len(U_reg)`).
    10. Count the number of GF sandwiches currently on trays (`avail_gf_ontray`) and Regular sandwiches currently on trays (`avail_reg_ontray`) in the state.
    11. Calculate how many GF sandwiches still need to be put on trays: `needing_put_on_tray_gf = max(0, needed_gf - avail_gf_ontray)`. Add this cost to `H`.
    12. Calculate how many Regular sandwiches still need to be put on trays: `needing_put_on_tray_reg = max(0, needed_reg - avail_reg_ontray)`. Add this cost to `H`.
    13. Count the number of GF sandwiches already made (either `at_kitchen_sandwich` or `ontray`) (`avail_gf_made`) and Regular sandwiches already made (`avail_reg_made`) in the state.
    14. Calculate how many GF sandwiches still need to be made: `needing_make_gf = max(0, needed_gf - avail_gf_made)`. Add the cost for making these sandwiches (3 actions each) to `H`. `H += needing_make_gf * 3`.
    15. Calculate how many Regular sandwiches still need to be made: `needing_make_reg = max(0, needed_reg - avail_reg_made)`. Add the cost for making these sandwiches (3 actions each) to `H`. `H += needing_make_reg * 3`.
    16. The total value of `H` is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children, allergy status,
        and waiting locations from the task definition.
        """
        super().__init__(task)

        # Extract children who are goals
        self.goal_children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "served":
                self.goal_children.add(parts[1])

        # Extract allergy status from static facts
        self.allergic_gluten = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "allergic_gluten":
                self.allergic_gluten.add(parts[1])

        # Extract waiting locations from static facts
        self.child_waiting_table = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "waiting":
                child, table = parts[1], parts[2]
                self.child_waiting_table[child] = table

    def is_sandwich_gf(self, sandwich_name, current_state):
        """
        Helper to check if a sandwich is gluten-free in the current state.
        Checks for either (is_gluten_free S) or (no_gluten_sandwich S).
        """
        return any(match(fact, "is_gluten_free", sandwich_name) for fact in current_state) or \
               any(match(fact, "no_gluten_sandwich", sandwich_name) for fact in current_state)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all unserved children.
        """
        state = node.state

        # 1. Identify unserved children
        unserved_children = {
            child for child in self.goal_children if f"(served {child})" not in state
        }

        # 2. If U is empty, the heuristic is 0
        if not unserved_children:
            return 0

        # 3. Initialize heuristic cost
        h = 0

        # 4. Cost for serving each child
        h += len(unserved_children)

        # 5. Identify tables with waiting unserved children
        tables_with_waiting_unserved = {
            self.child_waiting_table[child]
            for child in unserved_children
            if child in self.child_waiting_table # Should always be true for goal children in valid problems
        }

        # 6. Identify tables with trays
        tables_with_trays = {
            get_parts(fact)[2] for fact in state if match(fact, "at_table", "tray*", "*")
        }

        # 7. Count tables needing a tray move
        tables_needing_move = tables_with_waiting_unserved - tables_with_trays
        h += len(tables_needing_move)

        # 8. Separate unserved children by allergy status
        unserved_gf_children = {
            child for child in unserved_children if child in self.allergic_gluten
        }
        unserved_reg_children = unserved_children - unserved_gf_children

        # 9. Count needed sandwiches
        needed_gf = len(unserved_gf_children)
        needed_reg = len(unserved_reg_children)

        # 10. Count available sandwiches on trays
        avail_gf_ontray = sum(
            1 for fact in state if match(fact, "ontray", "sandw*", "tray*")
            and self.is_sandwich_gf(get_parts(fact)[1], state)
        )
        avail_reg_ontray = sum(
            1 for fact in state if match(fact, "ontray", "sandw*", "tray*")
            and not self.is_sandwich_gf(get_parts(fact)[1], state)
        )

        # 11. Calculate and add cost for putting on tray
        needing_put_on_tray_gf = max(0, needed_gf - avail_gf_ontray)
        needing_put_on_tray_reg = max(0, needed_reg - avail_reg_ontray)
        h += needing_put_on_tray_gf + needing_put_on_tray_reg

        # 13. Count available sandwiches already made (kitchen or on tray)
        # Sandwiches at kitchen: (at_kitchen_sandwich S)
        # Sandwiches on tray: (ontray S T)
        avail_gf_made = sum(
            1 for fact in state if (match(fact, "at_kitchen_sandwich", "sandw*") or match(fact, "ontray", "sandw*", "tray*"))
            and self.is_sandwich_gf(get_parts(fact)[1], state)
        )
        avail_reg_made = sum(
            1 for fact in state if (match(fact, "at_kitchen_sandwich", "sandw*") or match(fact, "ontray", "sandw*", "tray*"))
            and not self.is_sandwich_gf(get_parts(fact)[1], state)
        )

        # 14. Calculate and add cost for making GF sandwiches (3 actions each)
        needing_make_gf = max(0, needed_gf - avail_gf_made)
        h += needing_make_gf * 3

        # 15. Calculate and add cost for making Regular sandwiches (3 actions each)
        needing_make_reg = max(0, needed_reg - avail_reg_made)
        h += needing_make_reg * 3

        # 16. Return total heuristic value
        return h
