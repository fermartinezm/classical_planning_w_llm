from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases where fact might be empty or malformed, although unlikely with planner states
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate_name arg1 arg2)".
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

    Estimates the number of actions needed to serve all children.
    The heuristic counts the minimum required actions at each stage
    (make sandwich, put on tray, move tray, serve) for all unserved children,
    considering gluten requirements and available resources (sandwiches, trays).
    It is not admissible but aims to be efficiently computable and effective
    for greedy best-first search.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children's allergies.
        """
        # The goal conditions are available in task.goals, but this heuristic
        # focuses on the state of unserved children directly.
        # self.goals = task.goals

        static_facts = task.static

        # Extract allergy information from static facts for quick lookup
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children = {
             get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "not_allergic_gluten", "*")
        }
        # Note: Assumes every child is either allergic_gluten or not_allergic_gluten

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Pre-process state facts into a dictionary by predicate for efficient lookup
        state_facts_by_predicate = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty or malformed facts
                continue
            predicate = parts[0]
            if predicate not in state_facts_by_predicate:
                state_facts_by_predicate[predicate] = set()
            state_facts_by_predicate[predicate].add(fact)

        # Helper to check if a specific fact exists in the current state
        def has_predicate(predicate, *args):
             fact_str = "(" + " ".join([predicate] + list(args)) + ")"
             return predicate in state_facts_by_predicate and fact_str in state_facts_by_predicate[predicate]

        # 1. Identify unserved children and their locations
        served_children = {get_parts(fact)[1] for fact in state_facts_by_predicate.get('served', set())}
        waiting_children_at_loc = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in state_facts_by_predicate.get('waiting', set())
        }

        unserved_allergic_children_count = 0
        unserved_regular_children_count = 0
        locations_with_unserved_children = set()

        for child, loc in waiting_children_at_loc.items():
            if child not in served_children:
                locations_with_unserved_children.add(loc)
                if child in self.allergic_children:
                    unserved_allergic_children_count += 1
                elif child in self.not_allergic_children:
                     unserved_regular_children_count += 1
                # Children not in static info are ignored, assuming all relevant children are listed

        # If no unserved children, the goal is reached
        if unserved_allergic_children_count == 0 and unserved_regular_children_count == 0:
            return 0

        # 2. Count available sandwiches by type and location (kitchen/ontray)
        gf_sandwiches_kitchen_count = 0
        reg_sandwiches_kitchen_count = 0
        for fact in state_facts_by_predicate.get('at_kitchen_sandwich', set()):
            sandwich = get_parts(fact)[1]
            if has_predicate('no_gluten_sandwich', sandwich):
                gf_sandwiches_kitchen_count += 1
            else:
                reg_sandwiches_kitchen_count += 1

        gf_sandwiches_ontray_count = 0
        reg_sandwiches_ontray_count = 0
        for fact in state_facts_by_predicate.get('ontray', set()):
            sandwich = get_parts(fact)[1]
            if has_predicate('no_gluten_sandwich', sandwich):
                gf_sandwiches_ontray_count += 1
            else:
                reg_sandwiches_ontray_count += 1

        # 3. Count available ingredients and notexist sandwiches
        # These counts are primarily for understanding resource constraints,
        # but the heuristic sums action costs, assuming resources are available
        # for the calculated 'make' actions.
        gf_bread_count = len([b for b_fact in state_facts_by_predicate.get('at_kitchen_bread', set()) if has_predicate('no_gluten_bread', get_parts(b_fact)[1])])
        reg_bread_count = len([b for b_fact in state_facts_by_predicate.get('at_kitchen_bread', set()) if not has_predicate('no_gluten_bread', get_parts(b_fact)[1])])
        gf_content_count = len([c for c_fact in state_facts_by_predicate.get('at_kitchen_content', set()) if has_predicate('no_gluten_content', get_parts(c_fact)[1])])
        reg_content_count = len([c for c_fact in state_facts_by_predicate.get('at_kitchen_content', set()) if not has_predicate('no_gluten_content', get_parts(c_fact)[1])])
        notexist_sandwich_count = len(state_facts_by_predicate.get('notexist', set()))


        # 4. Count available trays by location
        tray_locations = {
            get_parts(fact)[2]
            for fact in state_facts_by_predicate.get('at', set())
            if get_parts(fact)[1].startswith('tray') # Ensure the object is a tray type
        }

        # 5. Calculate heuristic cost based on required actions at each stage

        total_cost = 0

        # Cost for 'serve' actions: One action is needed for each unserved child.
        total_cost += unserved_allergic_children_count + unserved_regular_children_count

        # Cost for 'put_on_tray' actions: For sandwiches that need to be on trays but aren't.
        # We need enough GF sandwiches on trays for allergic children, and enough total
        # sandwiches on trays for all children.
        needed_gf_on_trays = unserved_allergic_children_count
        needed_reg_on_trays = unserved_regular_children_count # These can be regular or GF

        # How many GF sandwiches need to be put on trays?
        needed_gf_put_on_tray = max(0, needed_gf_on_trays - gf_sandwiches_ontray_count)

        # How many Regular sandwiches need to be put on trays?
        # This is the total needed on trays minus the GF ones already on trays or needing to be put on trays.
        # Regular sandwiches can satisfy regular children. GF sandwiches can satisfy regular children too.
        # Let's count total sandwiches needed on trays: unserved_allergic + unserved_regular
        # Available on trays: gf_sandwiches_ontray_count + reg_sandwiches_ontray_count
        # Total needing put on tray: max(0, (unserved_allergic_children_count + unserved_regular_children_count) - (gf_sandwiches_ontray_count + reg_sandwiches_ontray_count))
        # This total need must be met by sandwiches at kitchen.
        # Let's count how many sandwiches (GF or Reg) need to transition from kitchen to ontray.
        # Total sandwiches needed on trays = unserved_allergic_children_count + unserved_regular_children_count
        # Total sandwiches currently on trays = gf_sandwiches_ontray_count + reg_sandwiches_ontray_count
        # Total sandwiches that need to be put on trays = max(0, (unserved_allergic_children_count + unserved_regular_children_count) - (gf_sandwiches_ontray_count + reg_sandwiches_ontray_count))
        # Cost += Total sandwiches that need to be put on trays

        # A simpler approach for put_on_tray: Count how many sandwiches *of each type* are needed on trays
        # vs. available on trays.
        # Needed GF on tray: unserved_allergic_children_count
        # Available GF on tray: gf_sandwiches_ontray_count
        # GF needing put on tray: max(0, unserved_allergic_children_count - gf_sandwiches_ontray_count)

        # Needed Regular on tray: unserved_regular_children_count
        # Available Regular on tray: reg_sandwiches_ontray_count
        # Regular needing put on tray: max(0, unserved_regular_children_count - reg_sandwiches_ontray_count)
        # This doesn't account for GF sandwiches serving regular children.

        # Let's refine:
        # Total sandwiches needed on trays = unserved_allergic_children_count + unserved_regular_children_count
        # Total sandwiches available on trays = gf_sandwiches_ontray_count + reg_sandwiches_ontray_count
        # Total put_on_tray actions needed = max(0, Total sandwiches needed on trays - Total sandwiches available on trays)
        # This is a lower bound on put_on_tray actions. It doesn't guarantee the *right type* is put on tray.

        # A better approach for put_on_tray cost:
        # Count GF sandwiches needed on trays: unserved_allergic_children_count
        # Count Regular sandwiches needed on trays: unserved_regular_children_count
        # Available GF on trays: gf_sandwiches_ontray_count
        # Available Regular on trays: reg_sandwiches_ontray_count
        # GF sandwiches that *must* be put on trays (cannot be substituted): max(0, unserved_allergic_children_count - gf_sandwiches_ontray_count)
        # Regular sandwiches needed on trays that can be Regular or remaining GF:
        # Remaining Regular needed = unserved_regular_children_count
        # Remaining GF available = max(0, gf_sandwiches_ontray_count - unserved_allergic_children_count) # GF on trays not needed by allergic
        # Remaining sandwiches needed on trays (can be Reg or GF) = max(0, remaining_regular_needed - remaining_GF_available)
        # This is getting complicated. Let's simplify for a non-admissible heuristic.

        # Simplified put_on_tray cost: Count how many sandwiches of each type are needed on trays
        # vs. available on trays, summing the deficits.
        needed_gf_put_on_tray = max(0, unserved_allergic_children_count - gf_sandwiches_ontray_count)
        needed_reg_put_on_tray = max(0, unserved_regular_children_count - reg_sandwiches_ontray_count) # This is for regular children, can use reg or surplus GF
        # Let's just count the total deficit of sandwiches on trays vs total needed.
        total_sandwiches_needed_on_trays = unserved_allergic_children_count + unserved_regular_children_count
        total_sandwiches_on_trays = gf_sandwiches_ontray_count + reg_sandwiches_ontray_count
        put_on_tray_actions_needed = max(0, total_sandwiches_needed_on_trays - total_sandwiches_on_trays)
        total_cost += put_on_tray_actions_needed


        # Cost for 'make_sandwich' actions: For sandwiches that need to be put on trays but aren't made yet.
        # These must come from sandwiches at the kitchen.
        # Total sandwiches needed at kitchen (to be put on trays) = put_on_tray_actions_needed
        # Total sandwiches available at kitchen = gf_sandwiches_kitchen_count + reg_sandwiches_kitchen_count
        # Total make actions needed = max(0, put_on_tray_actions_needed - total_sandwiches_available_at_kitchen)
        # This doesn't distinguish GF/Reg makes.

        # Let's distinguish makes by type:
        # GF sandwiches needed at kitchen (to be put on trays): needed_gf_put_on_tray
        # Available GF at kitchen: gf_sandwiches_kitchen_count
        # GF make actions needed: max(0, needed_gf_put_on_tray - gf_sandwiches_kitchen_count)

        # Regular sandwiches needed at kitchen (to be put on trays): needed_reg_put_on_tray
        # Available Regular at kitchen: reg_sandwiches_kitchen_count
        # Regular make actions needed: max(0, needed_reg_put_on_tray - reg_sandwiches_kitchen_count)
        # This still doesn't account for surplus GF at kitchen satisfying regular needs.

        # Let's use the total deficit approach again for makes:
        # Total sandwiches needed at kitchen (to eventually be put on trays and served)
        # = Total sandwiches needed on trays - Total sandwiches already on trays
        # = max(0, (unserved_allergic_children_count + unserved_regular_children_count) - (gf_sandwiches_ontray_count + reg_sandwiches_ontray_count))
        # Total sandwiches available at kitchen = gf_sandwiches_kitchen_count + reg_sandwiches_kitchen_count
        # Total make actions needed = max(0, max(0, (unserved_allergic_children_count + unserved_regular_children_count) - (gf_sandwiches_ontray_count + reg_sandwiches_ontray_count)) - (gf_sandwiches_kitchen_count + reg_sandwiches_kitchen_count))
        # This is simpler and captures the overall flow.

        total_sandwiches_needed_made = max(0, put_on_tray_actions_needed - (gf_sandwiches_kitchen_count + reg_sandwiches_kitchen_count))
        total_cost += total_sandwiches_needed_made


        # Cost for 'move_tray' actions: For locations needing a tray that don't have one.
        # A tray is needed at any location where an unserved child is waiting.
        # A tray is also needed at the kitchen if sandwiches need to be put on trays.
        required_tray_locations = set(locations_with_unserved_children)
        if put_on_tray_actions_needed > 0:
             required_tray_locations.add('kitchen')

        tray_move_cost = 0
        for loc in required_tray_locations:
            if loc not in tray_locations:
                tray_move_cost += 1

        total_cost += tray_move_cost

        # The heuristic does not explicitly count costs for insufficient ingredients
        # or 'notexist' sandwiches beyond the number of 'make' actions needed.
        # If ingredients/notexist are truly a bottleneck, the problem might be
        # unsolvable, but the heuristic returns a finite value, which is acceptable.

        return total_cost

