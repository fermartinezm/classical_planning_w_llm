from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Mock Heuristic base class for standalone testing if needed
class Heuristic:
    """
    Base class for domain-dependent heuristics.
    Subclasses should implement __init__(self, task) and __call__(self, node).
    """
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError


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
    if len(parts) != len(args):
        return False # Pattern length must match fact length
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the number of unserved children and adds estimated costs for
    getting a suitable sandwich to each child's table, considering the sandwich's
    current location (at table, on tray in kitchen, in kitchen, needs making)
    and the child's dietary needs (gluten allergy). It attempts to account for
    shared resources like trays and ingredients in a simplified way by counting
    the number of sandwiches needing to pass through each stage (make, put on tray, move tray).

    # Assumptions
    - Each child needs exactly one sandwich.
    - A tray holds exactly one sandwich at a time (inferred from `(ontray S T)` predicate structure).
    - Making a sandwich requires one bread and one content portion.
    - Gluten-free sandwiches require gluten-free bread and content.
    - Ingredients in the kitchen are the only source for new sandwiches.
    - Trays in the kitchen are the only source for trays to move sandwiches to tables.
    - The heuristic does not guarantee admissibility but aims for accuracy for GBFS.

    # Heuristic Initialization
    - Identify all children that need to be served from the goal state.
    - Identify which children are allergic to gluten from static facts.
    - Identify which bread and content portions are gluten-free from static facts.
    - Identify which bread and content portions are *not* gluten-free (assuming this is static).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all children who are waiting but not yet served.
    2. Separate these children into those who are allergic to gluten (need GF sandwiches)
       and those who are not (can use Any sandwich). Count how many of each type are needed.
    3. The base heuristic cost is the number of unserved children (representing the final 'serve' action for each).
    4. Count available sandwiches by type (GF/NonGF) and location (on tray at table, on tray in kitchen, in kitchen).
    5. Count available ingredients by type (GF/NonGF) in the kitchen.
    6. Calculate the deficit of suitable sandwiches needed at tables that are not currently there.
    7. For the deficit sandwiches, estimate additional actions needed to get them to the table,
       prioritizing sandwiches closer to the goal location:
       - Count how many needed sandwiches can come from trays in the kitchen (cost: +1 move action per sandwich/tray). Update the deficit.
       - Count how many remaining needed sandwiches can come from the kitchen (not on trays) (cost: +1 put-on-tray action per sandwich). Update the deficit.
       - Count how many remaining needed sandwiches must be made from ingredients (cost: +1 make action per sandwich). Update the deficit.
       - Ensure ingredient availability is checked when counting sandwiches that can be made. Prioritize GF ingredients for GF sandwiches.
    8. Sum the base cost and the estimated additional action costs.

    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children, allergy info,
        and gluten-free ingredient info.
        """
        self.goals = task.goals
        self.static = task.static

        # Identify all children that need to be served in the goal state
        self.goal_children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "served":
                self.goal_children.add(parts[1])

        # Identify which children are allergic based on static facts
        self.allergic_children = {
            get_parts(fact)[1] for fact in self.static if match(fact, "allergic_gluten", "*")
        }

        # Identify gluten-free ingredients based on static facts
        self.gf_bread = {
            get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_bread", "*")
        }
        self.gf_content = {
            get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_content", "*")
        }

        # Identify non-gluten-free ingredients based on initial state and static GF info
        # Assuming all bread/content objects are listed in the initial state and their GF status is static.
        all_bread_in_init = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "at_kitchen_bread", "*")}
        all_content_in_init = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "at_kitchen_content", "*")}
        self.nongf_bread = all_bread_in_init - self.gf_bread
        self.nongf_content = all_content_in_init - self.gf_content


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children and separate by allergy
        unserved_allergic = set()
        unserved_non_allergic = set()

        for child in self.goal_children:
            if f"(served {child})" not in state:
                if child in self.allergic_children:
                    unserved_allergic.add(child)
                else:
                    unserved_non_allergic.add(child)

        needed_gf = len(unserved_allergic)
        needed_any = len(unserved_non_allergic)

        # If no children need serving, heuristic is 0
        if needed_gf == 0 and needed_any == 0:
            return 0

        # 3. Base heuristic cost: 1 action per unserved child (the 'serve' action)
        h = needed_gf + needed_any

        # 4. Count available sandwiches by type and location
        s_table_gf = 0
        s_table_nongf = 0
        s_ontray_kitchen_gf = 0
        s_ontray_kitchen_nongf = 0
        s_kitchen_gf = 0
        s_kitchen_nongf = 0

        # Track which sandwiches are GF based on state facts or inferred from ingredients
        sandwich_is_gf = {}
        all_sandwiches_in_state = set()

        # Find sandwiches on trays and their locations
        sandwiches_on_trays = {} # Map sandwich -> tray
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1:3]
                sandwiches_on_trays[s] = t
                all_sandwiches_in_state.add(s)

        trays_at_location = {} # Map tray -> location
        for fact in state:
            if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:3]
                 # Only track trays and kitchen/table locations
                 if obj.startswith("tray") and (loc == "kitchen" or loc.startswith("table")):
                     trays_at_location[obj] = loc

        # Find sandwiches in the kitchen (not on trays)
        sandwiches_in_kitchen = set()
        for fact in state:
             if match(fact, "at_kitchen_sandwich", "*"):
                 s = get_parts(fact)[1]
                 sandwiches_in_kitchen.add(s)
                 all_sandwiches_in_state.add(s)

        # Determine GF status for all relevant sandwiches
        sandwich_ingredients = {} # Map sandwich -> {bread: B, content: Cnt}
        for fact in state:
             if match(fact, "has_bread", "*", "*"):
                 s, b = get_parts(fact)[1:3]
                 sandwich_ingredients.setdefault(s, {})['bread'] = b
             if match(fact, "has_content", "*", "*"):
                 s, c = get_parts(fact)[1:3]
                 sandwich_ingredients.setdefault(s, {})['content'] = c

        for s in all_sandwiches_in_state:
            # Check direct predicates first
            if f"(is_gluten_free {s})" in state or f"(no_gluten_sandwich {s})" in state:
                 sandwich_is_gf[s] = True
            # If not directly marked, infer from ingredients if available
            elif s in sandwich_ingredients:
                ingredients = sandwich_ingredients[s]
                bread = ingredients.get('bread')
                content = ingredients.get('content')
                is_gf = True
                if bread and bread not in self.gf_bread:
                    is_gf = False
                if content and content not in self.gf_content:
                    is_gf = False
                sandwich_is_gf[s] = is_gf
            else:
                 # If status and ingredients unknown, assume non-GF for safety
                 sandwich_is_gf[s] = False


        # Count sandwiches by location and type
        for s in all_sandwiches_in_state:
            is_gf = sandwich_is_gf.get(s, False) # Default to False if status couldn't be determined

            if s in sandwiches_on_trays:
                t = sandwiches_on_trays[s]
                loc = trays_at_location.get(t) # Tray must have a location

                if loc and loc.startswith("table"):
                    if is_gf:
                         s_table_gf += 1
                    else:
                         s_table_nongf += 1
                elif loc == "kitchen":
                    if is_gf:
                        s_ontray_kitchen_gf += 1
                    else:
                        s_ontray_kitchen_nongf += 1
            elif s in sandwiches_in_kitchen:
                 if is_gf:
                     s_kitchen_gf += 1
                 else:
                     s_kitchen_nongf += 1

        # Account for sandwiches already at tables (these fulfill needs without further action)
        # Allergic children *must* use GF sandwiches
        served_by_at_table_gf = min(needed_gf, s_table_gf)
        needed_gf -= served_by_at_table_gf

        # Non-allergic children can use any remaining sandwich at tables (GF or Non-GF)
        remaining_at_table_gf = s_table_gf - served_by_at_table_gf
        available_at_table_any = remaining_at_table_gf + s_table_nongf
        served_by_at_table_any = min(needed_any, available_at_table_any)
        needed_any -= served_by_at_table_any

        # Calculate deficit that needs to come from kitchen stages
        deficit_from_kitchen_gf = needed_gf
        deficit_from_kitchen_any = needed_any # These can be GF or Non-GF

        # 5. Count available suitable ingredients in the kitchen
        available_bread_kitchen_gf = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and get_parts(fact)[1] in self.gf_bread)
        available_content_kitchen_gf = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and get_parts(fact)[1] in self.gf_content)
        available_bread_kitchen_nongf = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and get_parts(fact)[1] in self.nongf_bread)
        available_content_kitchen_nongf = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and get_parts(fact)[1] in self.nongf_content)

        # 7. Calculate additional costs based on deficit and available resources

        # Cost for moving sandwiches from kitchen trays to tables (+1 per sandwich/tray)
        # Prioritize GF for allergic needs first
        move_from_ontray_kitchen_gf_for_allergic = min(deficit_from_kitchen_gf, s_ontray_kitchen_gf)
        deficit_from_kitchen_gf -= move_from_ontray_kitchen_gf_for_allergic
        h += move_from_ontray_kitchen_gf_for_allergic

        # Use remaining GF on trays or NonGF on trays for Any needs
        remaining_ontray_kitchen_gf = s_ontray_kitchen_gf - move_from_ontray_kitchen_gf_for_allergic
        move_from_ontray_kitchen_any = min(deficit_from_kitchen_any, s_ontray_kitchen_nongf + remaining_ontray_kitchen_gf)
        deficit_from_kitchen_any -= move_from_ontray_kitchen_any
        h += move_from_ontray_kitchen_any


        # Cost for putting sandwiches from kitchen onto trays (+1 per sandwich)
        # Prioritize GF for allergic needs first
        put_from_kitchen_gf_for_allergic = min(deficit_from_kitchen_gf, s_kitchen_gf)
        deficit_from_kitchen_gf -= put_from_kitchen_gf_for_allergic
        h += put_from_kitchen_gf_for_allergic

        # Use remaining GF in kitchen or NonGF in kitchen for Any needs
        remaining_kitchen_gf = s_kitchen_gf - put_from_kitchen_gf_for_allergic
        put_from_kitchen_any = min(deficit_from_kitchen_any, s_kitchen_nongf + remaining_kitchen_gf)
        deficit_from_kitchen_any -= put_from_kitchen_any
        h += put_from_kitchen_any


        # Cost for making new sandwiches (+1 per sandwich)
        # GF sandwiches must be made from GF ingredients
        make_gf = min(deficit_from_kitchen_gf, available_bread_kitchen_gf, available_content_kitchen_gf)
        deficit_from_kitchen_gf -= make_gf
        h += make_gf

        # Use remaining ingredients for Any sandwiches
        remaining_bread_gf = available_bread_kitchen_gf - make_gf
        remaining_content_gf = available_content_kitchen_gf - make_gf

        # Ingredients available for Any sandwiches (remaining GF + NonGF)
        available_bread_for_any = remaining_bread_gf + available_bread_kitchen_nongf
        available_content_for_any = remaining_content_gf + available_content_kitchen_nongf

        make_any = min(deficit_from_kitchen_any, available_bread_for_any, available_content_for_any)
        deficit_from_kitchen_any -= make_any
        h += make_any

        # Note: Any remaining deficit_from_kitchen_gf or deficit_from_kitchen_any means
        # we cannot make enough sandwiches with current ingredients. The heuristic
        # still returns a finite value, indicating remaining work that cannot be
        # completed with current kitchen stock.

        return h
