from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not running within the planner environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError


# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
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
    This heuristic estimates the number of actions required to serve all unserved children.
    It counts the number of sandwiches needed (gluten-free for allergic children, any for others)
    and estimates the cost based on the current state of available sandwiches and ingredients:
    sandwiches already on trays (cost 1), sandwiches in the kitchen (cost 3), or sandwiches
    that need to be made from ingredients (cost 4). It prioritizes using sandwiches that are
    closer to being served.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Sandwiches already on trays can be served with 1 action (ignoring tray location relative to child).
    - Sandwiches in the kitchen require 3 actions (put_on_tray, move_tray, serve). This assumes a tray is available at the kitchen and can be moved to the child's location.
    - Sandwiches made from ingredients require 4 actions (make, put_on_tray, move_tray, serve). This assumes ingredients, a sandwich object, and a tray are available.
    - Resource conflicts (like limited trays or specific tray locations) are relaxed. Ingredient availability for making is considered.
    - Gluten-free sandwiches can satisfy demand from both allergic and non-allergic children, but are prioritized for allergic children.

    # Heuristic Initialization
    The heuristic extracts static information about which children are allergic, which bread portions are gluten-free, and which content portions are gluten-free.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all unserved children by checking `waiting` facts against `served` facts. Categorize unserved children by allergy status (using static facts) to determine the total demand for gluten-free (`needed_gf`) and regular (`needed_reg`) sandwiches.
    2. Count the available sandwiches in the current state, categorized by location (`ontray`, `at_kitchen_sandwich`) and type (gluten-free, regular), using `no_gluten_sandwich` facts.
    3. Count available ingredients (`at_kitchen_bread`, `at_kitchen_content`) and `notexist` sandwich objects, categorized by gluten status using static facts.
    4. Calculate the total heuristic cost by satisfying the demand for sandwiches in phases, starting with the "closest" available sandwiches:
        a. **Phase 1 (Cost 1 per sandwich):** Use available sandwiches that are already `ontray`. Prioritize gluten-free sandwiches for allergic children first, then use any remaining `ontray` sandwiches (including surplus GF) for regular demand. Add 1 to the total cost for each sandwich used in this phase.
        b. **Phase 2 (Cost 3 per sandwich):** Use available sandwiches that are `at_kitchen_sandwich` for any remaining demand. Prioritize gluten-free kitchen sandwiches for remaining allergic demand, then use any remaining kitchen sandwiches (including surplus GF) for remaining regular demand. Add 3 to the total cost for each sandwich used in this phase.
        c. **Phase 3 (Cost 4 per sandwich):** For any remaining demand, calculate how many new sandwiches can be made from available ingredients and `notexist` objects. Prioritize making gluten-free sandwiches if needed and possible, consuming the required ingredients and objects. Then, make regular sandwiches if needed and possible, using any remaining ingredients (including surplus GF ingredients) and objects. Add 4 to the total cost for each sandwich that can be made.
    5. The total cost accumulated across all phases is the heuristic value. If all children are served, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        super().__init__(task) # Call the base class constructor

        # Extract static information for quick lookup
        self.static_allergic_children = {
            get_parts(fact)[1] for fact in self.static if match(fact, "allergic_gluten", "*")
        }
        self.static_not_allergic_children = {
            get_parts(fact)[1] for fact in self.static if match(fact, "not_allergic_gluten", "*")
        }
        self.static_no_gluten_bread = {
            get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_bread", "*")
        }
        self.static_no_gluten_content = {
            get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_content", "*")
        }

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # --- Step 1: Identify unserved children and demand ---
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        waiting_children = {get_parts(fact)[1] for fact in state if match(fact, "waiting", "*", "*")}

        unserved_children = waiting_children - served_children

        # If no children are waiting or all waiting children are served, goal is reached.
        if not unserved_children:
             return 0

        needed_gf = 0 # Demand for gluten-free sandwiches
        needed_reg = 0 # Demand for regular sandwiches

        for child in unserved_children:
            if child in self.static_allergic_children:
                needed_gf += 1
            elif child in self.static_not_allergic_children:
                needed_reg += 1
            # Assuming all waiting children are either allergic or not_allergic

        # --- Step 2 & 3: Count available resources ---
        at_kitchen_bread = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        at_kitchen_content = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}
        at_kitchen_sandwich = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        ontray_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "ontray", "*", "*")}
        no_gluten_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        notexist_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")}

        # Available sandwiches by location and type
        avail_ontray_gf = len([s for s in ontray_sandwiches if s in no_gluten_sandwiches_in_state])
        avail_ontray_reg = len([s for s in ontray_sandwiches if s not in no_gluten_sandwiches_in_state])

        avail_kitchen_gf = len([s for s in at_kitchen_sandwich if s in no_gluten_sandwiches_in_state])
        avail_kitchen_reg = len([s for s in at_kitchen_sandwich if s not in no_gluten_sandwiches_in_state])

        # Available ingredients and sandwich objects counts
        avail_gf_bread_count = len([b for b in at_kitchen_bread if b in self.static_no_gluten_bread])
        avail_gf_content_count = len([c for c in at_kitchen_content if c in self.static_no_gluten_content])
        # Regular bread/content is any bread/content that is not GF
        avail_reg_bread_count = len(at_kitchen_bread) - avail_gf_bread_count
        avail_reg_content_count = len(at_kitchen_content) - avail_gf_content_count
        avail_notexist_s_count = len(notexist_sandwiches)

        # --- Step 4: Calculate cost based on phased supply ---
        total_cost = 0

        # Phase 1: Use sandwiches already on trays (Cost 1: serve)
        # Prioritize GF for GF demand
        served_gf_ontray = min(needed_gf, avail_ontray_gf)
        total_cost += served_gf_ontray * 1
        needed_gf -= served_gf_ontray
        avail_ontray_gf_surplus = avail_ontray_gf - served_gf_ontray

        # Use Reg and surplus GF for Reg demand
        served_reg_ontray_reg = min(needed_reg, avail_ontray_reg)
        total_cost += served_reg_ontray_reg * 1
        needed_reg -= served_reg_ontray_reg

        served_reg_ontray_gf = min(needed_reg, avail_ontray_gf_surplus)
        total_cost += served_reg_ontray_gf * 1
        needed_reg -= served_reg_ontray_gf

        # Phase 2: Use sandwiches in kitchen (Cost 3: put_on_tray + move_tray + serve)
        # Prioritize GF for GF demand
        served_gf_kitchen = min(needed_gf, avail_kitchen_gf)
        total_cost += served_gf_kitchen * 3
        needed_gf -= served_gf_kitchen
        avail_kitchen_gf_surplus = avail_kitchen_gf - served_gf_kitchen

        # Use Reg and surplus GF for Reg demand
        served_reg_kitchen_reg = min(needed_reg, avail_kitchen_reg)
        total_cost += served_reg_kitchen_reg * 3
        needed_reg -= served_reg_kitchen_reg

        served_reg_kitchen_gf = min(needed_reg, avail_kitchen_gf_surplus)
        total_cost += served_reg_kitchen_gf * 3
        needed_reg -= served_reg_kitchen_gf

        # Phase 3: Make new sandwiches (Cost 4: make + put_on_tray + move_tray + serve)
        # Make GF sandwiches first if needed and possible
        can_make_gf = min(avail_gf_bread_count, avail_gf_content_count, avail_notexist_s_count)
        to_make_gf = min(needed_gf, can_make_gf)
        total_cost += to_make_gf * 4
        needed_gf -= to_make_gf
        # Consume ingredients/object counts for GF sandwiches
        avail_gf_bread_count -= to_make_gf
        avail_gf_content_count -= to_make_gf
        avail_notexist_s_count -= to_make_gf

        # Make Reg sandwiches if needed and possible, using remaining ingredients
        # Any remaining bread (GF or Reg) can be used for Reg sandwiches
        remaining_any_bread = avail_reg_bread_count + avail_gf_bread_count
        remaining_any_content = avail_reg_content_count + avail_gf_content_count

        can_make_reg = min(remaining_any_bread,
                           remaining_any_content,
                           avail_notexist_s_count)
        to_make_reg = min(needed_reg, can_make_reg)
        total_cost += to_make_reg * 4
        needed_reg -= to_make_reg
        # No need to update ingredient counts further as they are only used for making in this heuristic

        # Note: Any remaining needed_gf or needed_reg after Phase 3 indicates
        # that the required sandwiches cannot be produced with current resources.
        # The heuristic value reflects the cost for the sandwiches that *can* be provided.

        return total_cost
