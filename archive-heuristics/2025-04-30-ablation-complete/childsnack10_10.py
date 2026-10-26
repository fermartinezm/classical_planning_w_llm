from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It sums up the estimated costs for making necessary sandwiches, putting them on trays,
    moving trays to the children's locations, and finally serving the children.

    # Assumptions
    - Ingredients and 'notexist' sandwich objects are sufficient to make any required sandwiches.
    - Each make, put_on_tray, move_tray, and serve action costs 1.
    - The heuristic counts the total number of these actions needed across the problem,
      summing deficits in different stages (make, put, move, serve).
    - The cost of putting a sandwich on a tray is counted for sandwiches that are
      currently in the kitchen or will be newly made. The cost for existing kitchen
      sandwiches is only included if there are still children to be served.

    # Heuristic Initialization
    - Extracts static information: which children are allergic, and where each child is waiting.
    - Extracts goal information: the set of children who need to be served.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated as the sum of several components, each representing a type of action needed:

    1.  **Unserved Children Count:** Determine the number of children who are not yet served, categorized by allergy status (`U_gf`, `U_reg`). This directly contributes to the 'serve' action count.

    2.  **Sandwich Deficit Calculation:**
        - Count existing gluten-free sandwiches (`S_gf_made`: `at_kitchen_sandwich` or `ontray` with `no_gluten_sandwich`).
        - Count existing regular sandwiches (`S_reg_made`: `at_kitchen_sandwich` or `ontray` without `no_gluten_sandwich`).
        - Calculate the number of GF sandwiches that *must* be made (`needed_gf = max(0, U_gf - S_gf_made)`).
        - Calculate the number of Regular sandwiches that *must* be made (`needed_reg = max(0, U_reg - S_reg_made)`).
        - The total number of 'make' actions needed is `needed_gf + needed_reg`.

    3.  **Kitchen Sandwich Count:** Count sandwiches currently `(at_kitchen_sandwich s)` (`S_kitchen`). These need a 'put_on_tray' action if they are to be used.

    4.  **Locations Needing Tray:** Identify distinct locations where unserved children are waiting but no tray is currently present. The number of such locations (`len(locations_needing_tray)`) contributes to the 'move_tray' action count.

    5.  **Total Heuristic Calculation:**
        The heuristic sums the counts of necessary actions:
        - `(U_gf + U_reg)`: Each unserved child needs a 'serve' action.
        - `(needed_gf + needed_reg)`: Each sandwich needing creation needs a 'make' action.
        - `(needed_gf + needed_reg)`: Each newly made sandwich needs a 'put_on_tray' action.
        - `S_kitchen`: Each existing kitchen sandwich needs a 'put_on_tray' action. This cost is only included if there are still children to be served (`U_gf + U_reg > 0`).
        - `len(locations_needing_tray)`: Each location needing a tray needs a 'move_tray' action.

        Summing these gives:
        H = (U_gf + U_reg) + (needed_gf + needed_reg) + (needed_gf + needed_reg) + (S_kitchen if (U_gf + U_reg) > 0 else 0) + len(locations_needing_tray)

        Simplified:
        H = (U_gf + U_reg) + 2 * (needed_gf + needed_reg) + (S_kitchen if (U_gf + U_reg) > 0 else 0) + len(locations_needing_tray)

    This heuristic is 0 if and only if all children are served, as all component counts become zero in that state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information: allergic children and waiting locations
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "allergic_gluten", "*")
        }
        self.waiting_locations = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in static_facts
            if match(fact, "waiting", "*", "*")
        }
        # Extract all children from goals (assuming all children in goals are the ones to be served)
        self.all_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Count unserved children and categorize by allergy
        unserved_children = {c for c in self.all_children if f"(served {c})" not in state}
        U_gf = sum(1 for c in unserved_children if c in self.allergic_children)
        U_reg = len(unserved_children) - U_gf

        # If no children are unserved, the goal is reached, heuristic is 0.
        if U_gf == 0 and U_reg == 0:
            return 0

        # 2. Count existing sandwiches (made) and categorize by gluten-free status
        gluten_free_sandwiches_in_state = {
            s for fact in state if match(fact, "no_gluten_sandwich", "*") for s in get_parts(fact)[1:]
        }

        S_gf_made = 0 # GF sandwiches in kitchen or on tray
        S_reg_made = 0 # Reg sandwiches in kitchen or on tray
        S_kitchen = 0 # Any sandwich in kitchen

        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                s = get_parts(fact)[1]
                S_kitchen += 1
                if s in gluten_free_sandwiches_in_state:
                    S_gf_made += 1
                else:
                    S_reg_made += 1
            elif match(fact, "ontray", "*", "*"):
                 s = get_parts(fact)[1]
                 if s in gluten_free_sandwiches_in_state:
                    S_gf_made += 1
                 else:
                    S_reg_made += 1

        # 3. Calculate sandwiches needing to be made
        needed_gf = max(0, U_gf - S_gf_made)
        needed_reg = max(0, U_reg - S_reg_made)

        # 4. Identify locations needing a tray
        locations_with_unserved = {self.waiting_locations[c] for c in unserved_children}
        locations_with_trays = {
            p for fact in state if match(fact, "at", "*", "*") for t, p in [get_parts(fact)[1:]]
        }
        locations_needing_tray = locations_with_unserved - locations_with_trays

        # 5. Compute heuristic value based on the derived formula
        # H = (U_gf + U_reg) + 2 * (needed_gf + needed_reg) + (S_kitchen if (U_gf + U_reg) > 0 else 0) + len(locations_needing_tray)
        h_value = (U_gf + U_reg) + 2 * (needed_gf + needed_reg) + (S_kitchen if (U_gf + U_reg) > 0 else 0) + len(locations_needing_tray)

        return h_value
