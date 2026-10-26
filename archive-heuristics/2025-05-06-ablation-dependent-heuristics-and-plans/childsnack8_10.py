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
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simpler check is just element-wise matching up to the length of args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve all
    unserved children. It does this by counting the number of unserved children
    and the available resources (sandwiches in various states, ingredients)
    and greedily assigning the cheapest available resources/actions to satisfy
    the remaining serving tasks, prioritizing allergic children. The cost is
    calculated as the sum of the minimum actions needed for each unserved child,
    assuming resources can be allocated optimally across children (ignoring
    complex interactions like tray capacity and shared tray trips beyond simple counts).

    # Assumptions
    - Each child needs exactly one sandwich.
    - Allergic children require gluten-free sandwiches.
    - Non-allergic children can eat any sandwich.
    - The minimum steps to serve a child depend on the state of a suitable sandwich:
        - Sandwich on tray at child's location: 1 action (serve)
        - Sandwich on tray elsewhere: 2 actions (move tray, serve)
        - Sandwich in kitchen: 3 actions (put on tray, move tray, serve)
        - Sandwich needs to be made: 4 actions (make, put on tray, move tray, serve)
    - Resource counts (sandwiches, ingredients, notexist slots) are consumed greedily
      by the cheapest options first, prioritizing allergic children. Tray availability
      at the kitchen is implicitly handled by the 'put' action cost being tied to
      sandwiches originating from the kitchen or being made.

    # Heuristic Initialization
    - Extracts static information about children: their waiting place and allergy status.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all unserved children and their requirements (place, allergy).
    2. Count available resources in the current state:
       - Number of unserved allergic children.
       - Number of unserved non-allergic children.
       - Number of gluten-free (GF) and regular (Reg) sandwiches already on trays
         at places where unserved children are waiting.
       - Number of GF and Reg sandwiches already on trays elsewhere (not at required places).
       - Number of GF and Reg sandwiches in the kitchen.
       - Number of available 'notexist' sandwich slots.
       - Number of GF and Reg bread and content portions in the kitchen.
    3. Initialize total heuristic cost to 0.
    4. Calculate how many unserved children can be served by each source category
       (on-tray at place, on-tray elsewhere, kitchen, makable), prioritizing allergic
       children and cheaper sources.
       - For allergic children, consider GF sources in increasing order of cost (1, 2, 3, 4).
       - For non-allergic children, consider any remaining sources in increasing order of cost (1, 2, 3, 4).
       - When considering 'makable' sources (Cost 4), check available 'notexist' slots and ingredients.
    5. Sum the minimum costs for the number of children covered by each source category.
       Cost = (Children covered by Cost 1 source) * 1 + (Children covered by Cost 2 source) * 2 + ...
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        """
        # Store child info: {child_name: (place_name, is_allergic)}
        self.children_info = {}
        for fact in task.static:
            if match(fact, 'waiting', '*', '*'):
                child, place = get_parts(fact)[1:]
                # Find allergy status from static facts
                is_allergic = False
                for allergy_fact in task.static:
                    if match(allergy_fact, 'allergic_gluten', child):
                        is_allergic = True
                        break
                self.children_info[child] = (place, is_allergic)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        unserved_children = [] # list of (child_name, place_name, is_allergic)
        required_places = set()
        for child, (place, is_allergic) in self.children_info.items():
            if f'(served {child})' not in state:
                unserved_children.append((child, place, is_allergic))
                required_places.add(place)

        # If no unserved children, goal is reached
        if not unserved_children:
            return 0

        num_allergic_unserved = sum(is_allergic for _, _, is_allergic in unserved_children)
        num_non_allergic_unserved = len(unserved_children) - num_allergic_unserved

        # Helper to check if a sandwich is GF in the current state
        gf_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, 'no_gluten_sandwich', '*')}
        def is_sandwich_gf(sandwich_name):
             return sandwich_name in gf_sandwiches_in_state

        # Get tray locations
        tray_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, 'at', '*', '*')}

        # Count sandwiches by state and type
        num_gf_ontray_at_required_place = 0
        num_reg_ontray_at_required_place = 0
        num_gf_ontray_elsewhere = 0 # Includes trays at kitchen or other places
        num_reg_ontray_elsewhere = 0 # Includes trays at kitchen or other places

        for fact in state:
            if match(fact, 'ontray', '*', '*'):
                s, t = get_parts(fact)[1:]
                s_is_gf = is_sandwich_gf(s)
                t_loc = tray_locations.get(t)

                if t_loc in required_places:
                    if s_is_gf:
                        num_gf_ontray_at_required_place += 1
                    else:
                        num_reg_ontray_at_required_place += 1
                else: # Tray is at kitchen or some other place not currently needing a sandwich
                     if s_is_gf:
                        num_gf_ontray_elsewhere += 1
                     else:
                        num_reg_ontray_elsewhere += 1

        num_gf_kitchen = sum(1 for fact in state if match(fact, 'at_kitchen_sandwich', '*') and is_sandwich_gf(get_parts(fact)[1]))
        num_reg_kitchen = sum(1 for fact in state if match(fact, 'at_kitchen_sandwich', '*') and not is_sandwich_gf(get_parts(fact)[1]))

        num_notexist = sum(1 for fact in state if match(fact, 'notexist', '*'))

        num_gf_bread = sum(1 for fact in state if match(fact, 'at_kitchen_bread', '*') and match(fact, 'no_gluten_bread', get_parts(fact)[1]))
        num_reg_bread = sum(1 for fact in state if match(fact, 'at_kitchen_bread', '*') and not match(fact, 'no_gluten_bread', get_parts(fact)[1]))
        num_gf_content = sum(1 for fact in state if match(fact, 'at_kitchen_content', '*') and match(fact, 'no_gluten_content', get_parts(fact)[1]))
        num_reg_content = sum(1 for fact in state if match(fact, 'at_kitchen_content', '*') and not match(fact, 'no_gluten_content', get_parts(fact)[1]))

        # 3. Calculate needed sandwiches from each source, prioritizing allergic and cheaper sources

        # --- Allergic Children (Need GF) ---
        needed_gf_at_place = min(num_allergic_unserved, num_gf_ontray_at_required_place)
        rem_allergic = num_allergic_unserved - needed_gf_at_place

        needed_gf_elsewhere = min(rem_allergic, num_gf_ontray_elsewhere)
        rem_allergic -= needed_gf_elsewhere

        needed_gf_kitchen = min(rem_allergic, num_gf_kitchen)
        rem_allergic -= needed_gf_kitchen

        can_make_gf = min(num_notexist, num_gf_bread, num_gf_content)
        needed_gf_make = min(rem_allergic, can_make_gf)

        # Consume resources for GF making (for tracking remaining resources for Reg making)
        rem_notexist = num_notexist - needed_gf_make
        rem_gf_bread = num_gf_bread - needed_gf_make
        rem_gf_content = num_gf_content - needed_gf_make
        rem_reg_bread = num_reg_bread # Reg bread not used for GF
        rem_reg_content = num_reg_content # Reg content not used for GF


        # --- Non-Allergic Children (Can use Reg or GF) ---
        # Prioritize using remaining GF sandwiches first, then Reg.

        # Cost 1 (Any at place) - Use remaining GF at place, then Reg at place
        available_at_place = num_gf_ontray_at_required_place - needed_gf_at_place + num_reg_ontray_at_required_place
        needed_reg_at_place = min(num_non_allergic_unserved, available_at_place)
        rem_non_allergic = num_non_allergic_unserved - needed_reg_at_place

        # Cost 2 (Any on tray elsewhere) - Use remaining GF elsewhere, then Reg elsewhere
        available_elsewhere = num_gf_ontray_elsewhere - needed_gf_elsewhere + num_reg_ontray_elsewhere
        needed_reg_elsewhere = min(rem_non_allergic, available_elsewhere)
        rem_non_allergic -= needed_reg_elsewhere

        # Cost 3 (Any in kitchen) - Use remaining GF kitchen, then Reg kitchen
        available_kitchen = num_gf_kitchen - needed_gf_kitchen + num_reg_kitchen
        needed_reg_kitchen = min(rem_non_allergic, available_kitchen)
        rem_non_allergic -= needed_reg_kitchen

        # Cost 4 (Make Any - Reg or GF)
        # Can make Reg sandwiches using remaining notexist, any bread, any content
        rem_total_bread = rem_gf_bread + rem_reg_bread
        rem_total_content = rem_gf_content + rem_reg_content
        can_make_reg = min(rem_notexist, rem_total_bread, rem_total_content)
        needed_reg_make = min(rem_non_allergic, can_make_reg)
        # Note: Ingredient consumption for Reg making is not tracked in detail for simplicity.

        # 4. Calculate total cost
        total_cost = (needed_gf_at_place + needed_reg_at_place) * 1 \
                   + (needed_gf_elsewhere + needed_reg_elsewhere) * 2 \
                   + (needed_gf_kitchen + needed_reg_kitchen) * 3 \
                   + (needed_gf_make + needed_reg_make) * 4

        return total_cost
