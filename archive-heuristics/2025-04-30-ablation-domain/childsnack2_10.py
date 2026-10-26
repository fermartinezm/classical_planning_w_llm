from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this is available

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
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
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    who are currently waiting and not yet served. It counts the necessary actions
    for making sandwiches, putting them on trays, moving trays to tables, and serving.

    # Assumptions
    - Ingredients (bread, content) are available in the kitchen if they exist as objects.
    - Trays are available if they exist as objects.
    - Tray capacity is sufficient.
    - The number of sandwich objects is sufficient to make all needed sandwiches.
    - Robot movement within the kitchen or between the kitchen and tables is abstracted
      into tray movement cost.

    # Heuristic Initialization
    - Extract static information about children's gluten allergies to quickly check
      allergy status during heuristic computation.
    - Note: Static information about gluten-free ingredients is not directly used
      in the heuristic calculation, which relies on the `(no_gluten_sandwich ?s)`
      predicate being present in the state for existing GF sandwiches.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all children who are currently not served by checking for the absence
       of the `(served ?child)` predicate.
    2. Separate the unserved children into those who are allergic to gluten and
       those who are not, using the pre-extracted static allergy information.
    3. Count the number of complete sandwiches that already exist in the state
       (`(is_complete ?s)`), distinguishing between gluten-free (`(no_gluten_sandwich ?s)`)
       and non-gluten-free complete sandwiches.
    4. Calculate the number of gluten-free sandwiches that still need to be made.
       This is the number of unserved allergic children minus the count of
       available complete gluten-free sandwiches, capped at zero if sufficient
       GF sandwiches exist.
    5. Calculate the number of non-allergic sandwiches that still need to be made.
       This is the number of unserved non-allergic children minus the count of
       available complete non-gluten-free sandwiches and any excess available
       complete gluten-free sandwiches (i.e., GF sandwiches not needed by allergic
       children), capped at zero.
    6. Sum the counts from steps 4 and 5 to get the total number of sandwiches
       that need to be made from scratch (`S_make_total`).
    7. Count the number of complete sandwiches that are currently located on a tray
       (`(ontray ?s ?tray)` and `(is_complete ?s)`).
    8. Identify all tables where children are currently waiting (`(waiting ?child ?table)`).
    9. Identify all tables where a tray is currently located (`(at ?tray ?table)`).
       Exclude the kitchen location.
    10. Count the number of tables identified in step 8 that are not identified in step 9.
        These tables need a tray moved to them (`N_tables_need_tray`).
    11. The heuristic value is the sum of estimated costs for distinct action types:
        - Cost for making sandwiches: `S_make_total` (one 'make' action per sandwich).
        - Cost for putting sandwiches on trays: The number of unserved children
          minus the number of complete sandwiches already on trays, capped at zero
          (one 'put on tray' action for each needed sandwich not already on a tray).
        - Cost for moving trays: `N_tables_need_tray` (one 'move tray' action per table needing a tray).
        - Cost for serving children: The total number of unserved children
          (one 'serve' action per child).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts, specifically
        children's allergy statuses.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are facts that do not change during planning.
        self.static = task.static

        # Extract allergy status for each child from static facts
        self.allergic_children = set()
        self.not_allergic_children = set()
        # Assuming all children are listed in static facts with their allergy status
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 2:
                pred, child = parts
                if pred == "allergic_gluten":
                    self.allergic_children.add(child)
                elif pred == "not_allergic_gluten":
                    self.not_allergic_children.add(child)

        # We don't need to pre-extract GF bread/content as we rely on (no_gluten_sandwich ?s)
        # in the state for existing sandwiches.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        # Get all children known from static facts
        all_children = self.allergic_children | self.not_allergic_children
        # Get children who are currently served
        served_children = {
            get_parts(fact)[1] for fact in state if match(fact, "served", "*")
        }
        # Unserved children are those known children who are not served
        unserved_children = all_children - served_children

        # If no children are unserved, the goal is reached.
        if not unserved_children:
            return 0

        # 2. Separate unserved children by allergy status
        unserved_allergic = unserved_children.intersection(self.allergic_children)
        unserved_non_allergic = unserved_children.intersection(self.not_allergic_children)

        N_unserved = len(unserved_children)
        N_allergic_unserved = len(unserved_allergic)
        N_non_allergic_unserved = len(unserved_non_allergic)

        # 3. Count available complete sandwiches (total and GF)
        complete_sandwiches = set()
        gf_complete_sandwiches = set()
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == "is_complete":
                sandwich = parts[1]
                complete_sandwiches.add(sandwich)
                # Check if this complete sandwich is gluten-free using the predicate
                is_gf_fact = f"(no_gluten_sandwich {sandwich})"
                if is_gf_fact in state:
                     gf_complete_sandwiches.add(sandwich)

        C_complete = len(complete_sandwiches)
        C_gf_complete = len(gf_complete_sandwiches)
        C_non_gf_complete = C_complete - C_gf_complete

        # 4. Calculate GF sandwiches to make
        # Need N_allergic_unserved GF sandwiches. Have C_gf_complete.
        S_make_gf = max(0, N_allergic_unserved - C_gf_complete)

        # 5. Calculate non-allergic sandwiches to make
        # Need N_non_allergic_unserved non-allergic sandwiches.
        # Can use available non-GF complete sandwiches (C_non_gf_complete).
        # Can also use excess available GF complete sandwiches.
        excess_gf_complete = max(0, C_gf_complete - N_allergic_unserved)
        available_for_non_allergic = C_non_gf_complete + excess_gf_complete
        S_make_non_gf = max(0, N_non_allergic_unserved - available_for_non_allergic)

        # 6. Total sandwiches to make
        S_make_total = S_make_gf + S_make_non_gf

        # 7. Count complete sandwiches already on trays
        complete_sandwiches_on_trays = set()
        for fact in state:
            parts = get_parts(fact)
            # Check for (ontray ?s ?tray)
            if len(parts) == 3 and parts[0] == "ontray":
                sandwich = parts[1]
                # Check if this sandwich is also complete
                is_complete_fact = f"(is_complete {sandwich})"
                if is_complete_fact in state:
                    complete_sandwiches_on_trays.add(sandwich)

        C_complete_ontray = len(complete_sandwiches_on_trays)

        # 8. Identify tables with waiting children
        tables_with_waiting_children = {
            get_parts(fact)[2] for fact in state if match(fact, "waiting", "*", "*")
        }

        # 9. Identify tables with trays (excluding kitchen)
        tables_with_trays = {
            get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("tray") # Assuming tray objects start with "tray"
        }
        tables_with_trays_at_tables = {t for t in tables_with_trays if t != 'kitchen'}

        # 10. Count tables needing a tray
        N_tables_need_tray = len(tables_with_waiting_children - tables_with_trays_at_tables)

        # 11. Calculate total heuristic cost
        # Cost = (Make actions) + (Put on tray actions) + (Move tray actions) + (Serve actions)
        # Assuming 1 action per step for simplicity.
        cost_make = S_make_total
        # Only count 'put on tray' for sandwiches that are needed but not already on a tray
        # We need N_unserved sandwiches in total. C_complete_ontray are already on trays.
        # The remaining N_unserved - C_complete_ontray needed sandwiches must be put on trays.
        cost_put_on_tray = max(0, N_unserved - C_complete_ontray)
        cost_move_tray = N_tables_need_tray
        cost_serve = N_unserved

        total_cost = cost_make + cost_put_on_tray + cost_move_tray + cost_serve

        return total_cost
