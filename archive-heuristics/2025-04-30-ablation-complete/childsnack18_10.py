from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
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
    It counts the number of tasks remaining in the snack delivery pipeline:
    making needed sandwiches, putting sandwiches onto trays, moving trays to
    children's locations, and finally serving the children.

    # Assumptions
    - Each child needs exactly one sandwich.
    - Allergic children require gluten-free sandwiches.
    - Non-allergic children can accept any sandwich.
    - Sandwiches must be on a tray at the child's location to be served.
    - There are enough ingredients, sandwich objects, and trays in the problem
      instance to satisfy the goal (assuming the problem is solvable).
    - The cost of each action is 1.

    # Heuristic Initialization
    - Identify all children that need to be served from the goal state.
    - Identify allergic and non-allergic children from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is calculated as follows:

    1.  **Count Unserved Children:** Determine how many children still need to be served based on the goal facts and the current state. If this count is zero, the heuristic is zero.
    2.  **Categorize Unserved Children:** Separate unserved children into allergic (needing GF sandwiches) and non-allergic (needing any sandwich).
    3.  **Count Available Sandwiches:** Count existing gluten-free and regular sandwiches, both at the kitchen and on trays, by inspecting the state facts. Also, identify which sandwich objects are gluten-free based on the `no_gluten_sandwich` predicate in the state.
    4.  **Estimate Sandwiches to Make:** Calculate how many new gluten-free and regular sandwiches must be made to satisfy the needs of unserved children, considering the available sandwiches. This contributes to the heuristic cost (1 action per sandwich made).
        - Number of GF sandwiches needed to make = `max(0, N_allergic_unserved - Avail_gf_total)`
        - Number of Regular sandwiches needed to make (for non-allergic children, after using available Reg and excess GF) = `max(0, N_non_allergic_unserved - Avail_reg_total - max(0, Avail_gf_total - N_allergic_unserved))`
        - Total sandwiches to Make = (Number of GF sandwiches needed to make) + (Number of Regular sandwiches needed to make)
    5.  **Estimate Sandwiches to Put on Trays:** Calculate how many sandwiches (existing at the kitchen or newly made) need to be moved onto trays. This is the total number of sandwiches needed for unserved children minus those already on trays. This contributes to the heuristic cost (1 action per sandwich put on tray).
        - Total Needed on Trays = Unserved Allergic + Unserved Non-Allergic
        - Total Already on Trays = Available GF on Tray + Available Reg on Tray
        - To Put on Tray = max(0, Total Needed on Trays - Total Already on Trays)
    6.  **Estimate Trays to Move:** Identify all places where unserved children are waiting (from the state). Identify all places where trays are currently located (from the state). Count the number of distinct places with waiting unserved children that do *not* currently have a tray. Each such place requires at least one tray movement. This contributes to the heuristic cost (1 action per tray movement to a new location).
    7.  **Estimate Serve Actions:** The number of serve actions required is simply the total number of unserved children. This contributes to the heuristic cost (1 action per child served).
    8.  **Sum Costs:** The total heuristic value is the sum of the estimated costs from steps 4, 5, 6, and 7.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Identify children who need to be served (from goals)
        # Goal is (served childX) for all children X that need serving.
        self.goal_children = {get_parts(g)[1] for g in self.goals if match(g, "served", "*")}

        # Identify allergic and non-allergic children (from static facts)
        self.allergic_children = {get_parts(f)[1] for f in self.static_facts if match(f, "allergic_gluten", "*")}
        self.non_allergic_children = {get_parts(f)[1] for f in self.static_facts if match(f, "not_allergic_gluten", "*")}

        # We don't need to store initial waiting places or tray locations
        # as these change and must be read from the current state.
        # We also don't need to store initial ingredient/sandwich counts
        # as the heuristic estimates actions based on *needed* vs *available*
        # in the current state, assuming solvability implies eventual availability.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state

        # 1. Count Unserved Children
        # A child is unserved if they are in the goal_children set but the (served child) fact is not in the state.
        unserved_children = {c for c in self.goal_children if f"(served {c})" not in state}
        N_unserved = len(unserved_children)

        # If all children are served, the goal is reached.
        if N_unserved == 0:
            return 0

        # 2. Categorize Unserved Children
        unserved_allergic = {c for c in unserved_children if c in self.allergic_children}
        unserved_non_allergic = {c for c in unserved_children if c in self.non_allergic_children}
        N_allergic_unserved = len(unserved_allergic)
        N_non_allergic_unserved = len(unserved_non_allergic)

        # 3. Count Available Sandwiches (and identify which are GF)
        # We need to know which sandwich objects are gluten-free.
        # This predicate is added by make_sandwich_no_gluten and is part of the state.
        gf_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        # Count sandwiches at the kitchen
        avail_gf_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*") and get_parts(fact)[1] in gf_sandwiches_in_state)
        avail_reg_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*") and get_parts(fact)[1] not in gf_sandwiches_in_state)

        # Count sandwiches on trays
        avail_gf_ontray = sum(1 for fact in state if match(fact, "ontray", "*", "*") and get_parts(fact)[1] in gf_sandwiches_in_state)
        avail_reg_ontray = sum(1 for fact in state if match(fact, "ontray", "*", "*") and get_parts(fact)[1] not in gf_sandwiches_in_state)

        Avail_gf_total = avail_gf_kitchen + avail_gf_ontray
        Avail_reg_total = avail_reg_kitchen + avail_reg_ontray

        # 4. Estimate Sandwiches to Make
        # Allergic children *must* have GF. Non-allergic can have GF or Reg.
        # Prioritize using available GF for allergic children.
        # Then use remaining available GF and all available Reg for non-allergic.
        # The number of sandwiches we need to *make* is the deficit after using existing ones.

        # Number of GF sandwiches we still need to make (for allergic children not covered by existing GF)
        Needed_gf_to_make = max(0, N_allergic_unserved - Avail_gf_total)

        # Number of Any sandwiches we still need to make for non-allergic children.
        # These are non-allergic children not covered by existing Reg sandwiches,
        # AND not covered by existing GF sandwiches that weren't needed by allergic children.
        remaining_non_allergic_needing_sandwich = max(0, N_non_allergic_unserved - Avail_reg_total - max(0, Avail_gf_total - N_allergic_unserved))
        Needed_reg_to_make = max(0, remaining_non_allergic_needing_sandwich) # Assume we make regular if possible

        To_make_sandw = Needed_gf_to_make + Needed_reg_to_make
        cost_make_sandw = To_make_sandw # Each make action costs 1

        # 5. Estimate Sandwiches to Put on Trays
        # Total sandwiches needed on trays is the total number of unserved children.
        Total_sandwiches_needed_on_trays = N_allergic_unserved + N_non_allergic_unserved
        Total_sandwiches_already_on_trays = avail_gf_ontray + avail_reg_ontray

        # Sandwiches that need to transition from kitchen/creation to being on a tray.
        # This count represents the number of 'put_on_tray' actions needed.
        # It's the total number of sandwiches required on trays minus those already there.
        To_put_on_tray = max(0, Total_sandwiches_needed_on_trays - Total_sandwiches_already_on_trays)
        cost_put_on_tray = To_put_on_tray # Each put_on_tray action costs 1

        # 6. Estimate Trays to Move
        # Identify places where unserved children are waiting.
        # The 'waiting' predicate is in the state.
        places_with_waiting_children = {get_parts(fact)[2] for fact in state if match(fact, "waiting", "*", "*") and get_parts(fact)[1] in unserved_children}

        # Identify places where trays are currently located.
        # The 'at' predicate for trays is in the state.
        places_with_trays = {get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("tray")}

        # Count places with waiting unserved children that do not have a tray.
        # Each such place requires at least one tray movement action to bring a tray there.
        Num_places_needing_tray_moved = len(places_with_waiting_children - places_with_trays)
        cost_move_tray = Num_places_needing_tray_moved # Each move_tray action costs 1

        # 7. Estimate Serve Actions
        # Each unserved child needs one serve action.
        cost_serve = N_unserved # Each serve action costs 1

        # Total heuristic is the sum of estimated costs for each stage.
        total_heuristic = cost_make_sandw + cost_put_on_tray + cost_move_tray + cost_serve

        return total_heuristic
