from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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
    # Ensure fact has at least as many parts as args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting
    children. It counts the number of unserved children (representing serve actions),
    the number of sandwiches that need to be made (representing make actions),
    the number of sandwiches that need to be moved from the kitchen onto a tray
    (representing put_on_tray actions), and the number of places where children
    are waiting that currently lack a tray (representing move_tray actions).

    # Assumptions
    - Each action (make, put on tray, move tray, serve) costs 1.
    - There are enough bread/content portions and 'notexist' sandwich objects
      available in the kitchen to make all necessary sandwiches.
    - Trays have unlimited capacity for sandwiches.
    - A tray is available somewhere to be moved to a needed location.
    - The heuristic counts the *number* of items/tasks that need to pass through
      different stages of the serving pipeline (make -> kitchen -> on_tray -> at_place -> served).

    # Heuristic Initialization
    - Identify all children that need to be served (from goal).
    - Identify which children are allergic to gluten (from static facts).
    - Identify which bread and content portions are gluten-free (from static facts).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic sums up the estimated number of actions needed for different
    stages of the process for all unserved children:

    1.  **Count Unserved Children (Serve Actions):** Iterate through the goal
        facts `(served ?c)`. If `(served ?c)` is not in the current state,
        child `?c` is unserved. The number of unserved children is a lower bound
        on the number of `serve` actions needed. Add this count to the heuristic.

    2.  **Count Sandwiches to Make (Make Actions):** Determine the number of
        gluten-free and regular sandwiches required for the unserved children.
        Count the available gluten-free and regular sandwiches currently in the
        kitchen or on trays. Calculate how many sandwiches of each type still
        need to be made to satisfy the demand, prioritizing available gluten-free
        sandwiches for allergic children first, then using any leftover GF
        sandwiches for non-allergic children's needs before considering regular
        sandwiches. The total number of sandwiches that need to be made is added
        to the heuristic.

    3.  **Count Sandwiches to Put on Tray (Put_on_tray Actions):** Sandwiches
        that are currently `at_kitchen_sandwich` or will be made need to be
        put on a tray before they can be served. Count the total number of
        sandwiches that are currently in the kitchen (`at_kitchen_sandwich`)
        plus the number of sandwiches estimated to be made (`N_make`).
        Count the number of sandwiches already on trays (`ontray`).
        The number of sandwiches that need to end up on trays is the total
        number of unserved children minus those already on trays.
        The number of `put_on_tray` actions needed is the minimum of
        (sandwiches in kitchen + sandwiches to be made) and (sandwiches needed on trays).
        Add this count to the heuristic.

    4.  **Count Tray Movements (Move_tray Actions):** Identify all places where
        unserved children are waiting. Identify all places where trays are currently
        located. For each place where unserved children are waiting, if no tray
        is currently at that place, a tray needs to be moved there. Count the
        number of such places. Add this count to the heuristic.

    The total heuristic value is the sum of the counts from steps 1, 2, 3, and 4.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children
        (allergy status) and ingredients (gluten-free status).
        """
        self.goals = task.goals # Goal conditions, used to find all children
        static_facts = task.static # Facts that are not affected by actions.

        # Identify all children that need to be served (from goal)
        self.all_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Identify which children are allergic or not (from static facts)
        self.allergic_children = {get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")}
        self.not_allergic_children = {get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")}

        # Identify which bread and content portions are gluten-free (from static facts)
        self.gf_bread = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")}
        self.gf_content = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")}

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state # Current world state.

        # 1. Count Unserved Children (Serve Actions)
        unserved_children = {c for c in self.all_children if f"(served {c})" not in state}
        N_unserved = len(unserved_children)

        # If all children are served, the goal is reached.
        if N_unserved == 0:
            return 0

        # Determine needed sandwich types for unserved children
        N_gf_needed = sum(1 for c in unserved_children if c in self.allergic_children)
        N_reg_needed = N_unserved - N_gf_needed # Children are either allergic or not

        # 2. Count Sandwiches to Make (Make Actions)
        # Count available sandwiches by type and location
        S_gf_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*") and f"(no_gluten_sandwich {get_parts(fact)[1]})" in state)
        S_reg_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*") and f"(no_gluten_sandwich {get_parts(fact)[1]})" not in state)
        S_gf_ontray = sum(1 for fact in state if match(fact, "ontray", "*", "*") and f"(no_gluten_sandwich {get_parts(fact)[1]})" in state)
        S_reg_ontray = sum(1 for fact in state if match(fact, "ontray", "*", "*") and f"(no_gluten_sandwich {get_parts(fact)[1]})" not in state)

        S_kitchen = S_gf_kitchen + S_reg_kitchen
        S_ontray = S_gf_ontray + S_reg_ontray
        S_gf_avail = S_gf_kitchen + S_gf_ontray
        S_reg_avail = S_reg_kitchen + S_reg_ontray

        # Calculate sandwiches to make based on needs and availability
        # GF sandwiches needed that must be made
        gf_satisfied_by_avail_gf = min(N_gf_needed, S_gf_avail)
        N_make_gf = N_gf_needed - gf_satisfied_by_avail_gf

        # GF sandwiches available after satisfying GF needs
        gf_leftover = max(0, S_gf_avail - N_gf_needed)

        # Regular sandwiches needed that must be made (after using available Reg and leftover GF)
        N_make_reg = max(0, N_reg_needed - (S_reg_avail + gf_leftover))

        N_make = N_make_gf + N_make_reg

        # 3. Count Sandwiches to Put on Tray (Put_on_tray Actions)
        # Total sandwiches needed on trays is the number of unserved children minus those already on trays
        N_need_ontray = max(0, N_unserved - S_ontray)

        # Sandwiches that will pass through the kitchen state (currently there or will be made)
        S_kitchen_plus_make = S_kitchen + N_make

        # Number of put_on_tray actions needed is limited by available kitchen sandwiches + those to be made,
        # and also by the number of sandwiches that actually need to end up on trays.
        N_put_on_tray = min(S_kitchen_plus_make, N_need_ontray)


        # 4. Count Tray Movements (Move_tray Actions)
        # Identify places where unserved children are waiting
        P_waiting = {get_parts(fact)[2] for fact in state if match(fact, "waiting", "*", "*") and get_parts(fact)[1] in unserved_children}

        # Identify places where trays are located
        P_trays = {get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")}

        # Count places with waiting children that do not have a tray
        N_places_need_tray_moved = len(P_waiting - P_trays)

        # Total heuristic is the sum of estimated actions for each stage
        total_cost = N_unserved + N_make + N_put_on_tray + N_places_need_tray_moved

        return total_cost
