from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Using math.inf for large penalty

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, although PDDL states are usually consistent
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
    This heuristic estimates the number of actions required to serve all unserved children.
    It sums up the estimated costs for four main types of actions needed:
    1. Serving each unserved child.
    2. Moving trays to places where children are waiting.
    3. Putting sandwiches onto trays (for those in the kitchen or yet to be made).
    4. Making sandwiches that do not yet exist but are needed.
    It adds a large penalty if needed sandwiches cannot be made due to missing
    'notexist' sandwich objects or missing ingredients of the required type.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - A tray can hold multiple sandwiches.
    - A tray can be moved to any place.
    - Ingredient resources (bread, content) are static and only available if present
      in the initial state (static facts).
    - The primary constraint on making sandwiches is the availability of 'notexist'
      sandwich objects and the required ingredients (GF or regular).

    # Heuristic Initialization
    - Extracts the goal conditions (which children need to be served).
    - Extracts static facts to determine which children are allergic/not allergic
      and where they are waiting.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all unserved children from the goal state and current state. If none, heuristic is 0.
    2. For each unserved child, determine their waiting place and allergy status using static facts.
    3. Calculate the base heuristic cost for 'serve' actions: 1 action for each unserved child.
    4. Identify all unique places where unserved children are waiting.
    5. Count trays currently located at these needed places.
    6. Estimate the number of 'move_tray' actions needed: the number of places needing service minus the number of trays already there (minimum 1 tray per needed place). Add this count to the heuristic.
    7. Determine if there are any unserved allergic or non-allergic children to know which types of sandwiches are 'suitable'.
    8. Count suitable sandwiches available anywhere (in kitchen or on trays) and specifically those in the kitchen. A sandwich is suitable if it's gluten-free (for allergic needs) or any type (for non-allergic needs).
    9. Calculate the number of suitable sandwiches that still need to be sourced (made), which is the total number of unserved children minus the number of suitable sandwiches already available anywhere.
    10. Count available 'notexist' sandwich objects.
    11. Calculate the number of sandwiches that can actually be made, limited by 'notexist' objects.
    12. Calculate the cost for 'make_sandwich' actions: 1 per sandwich that can be made. Add this to the heuristic.
    13. Calculate the cost for 'put_on_tray' actions: 1 for each suitable sandwich in the kitchen plus 1 for each sandwich that is made (as made sandwiches appear in the kitchen). Add this total to the heuristic.
    14. Calculate the number of GF and Regular sandwiches specifically needed to be made based on the deficit of available suitable sandwiches of each type.
    15. Check ingredient availability (any bread/content, GF bread/content) from the state.
    16. Add a large penalty if needed sandwiches cannot be made due to missing 'notexist' objects or missing ingredients of the required type (GF or regular).
    17. Sum all calculated costs and penalties.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information: waiting places and allergy status for all children
        self.waiting_places = {}  # child -> place
        self.allergy_status = {}  # child -> bool (is_allergic)
        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:]
                self.waiting_places[child] = place
            elif match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.allergy_status[child] = True
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.allergy_status[child] = False

        # Identify all children mentioned in static facts (assuming these are all children in the problem)
        self.all_children = set(self.waiting_places.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        unserved_children = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                child = get_parts(goal)[1]
                if goal not in state:
                    unserved_children.add(child)

        if not unserved_children:
            return 0  # Goal state reached

        h = 0

        # 3. Base cost for serve action
        num_unserved = len(unserved_children)
        h += num_unserved

        # 4. Identify places needing service
        places_needing_service = set(self.waiting_places[c] for c in unserved_children)

        # 5. Count trays at needed places
        trays_at_needed_places = set()
        tray_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, place = get_parts(fact)[1:]
                # Assuming only trays have 'at' predicate other than kitchen constant
                if obj.startswith("tray"):
                    tray_locations[obj] = place

        for t, p in tray_locations.items():
            if p in places_needing_service:
                trays_at_needed_places.add(t)

        # 6. Estimate move_tray actions needed
        num_tray_moves_needed = max(0, len(places_needing_service) - len(trays_at_needed_places))
        h += num_tray_moves_needed

        # 7. Determine suitability criteria
        has_unserved_allergic = any(self.allergy_status.get(c, False) for c in unserved_children)
        has_unserved_regular = any(not self.allergy_status.get(c, True) for c in unserved_children) # Default to allergic if status unknown

        # 8. Count suitable sandwiches available anywhere and in kitchen
        suitable_sandwiches_kitchen = set()
        suitable_sandwiches_anywhere = set()
        available_gf_anywhere = set() # GF sandwiches anywhere
        available_reg_anywhere = set() # Non-GF regular sandwiches anywhere

        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                s = get_parts(fact)[1]
                s_is_gluten_free = "(no_gluten_sandwich " + s + ")" in state
                is_suitable_for_any_unserved = (has_unserved_allergic and s_is_gluten_free) or has_unserved_regular
                if is_suitable_for_any_unserved:
                    suitable_sandwiches_kitchen.add(s)
                    suitable_sandwiches_anywhere.add(s)
                if s_is_gluten_free:
                    available_gf_anywhere.add(s)
                else:
                    available_reg_anywhere.add(s)

            elif match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1:]
                s_is_gluten_free = "(no_gluten_sandwich " + s + ")" in state
                is_suitable_for_any_unserved = (has_unserved_allergic and s_is_gluten_free) or has_unserved_regular
                if is_suitable_for_any_unserved:
                    suitable_sandwiches_anywhere.add(s)
                if s_is_gluten_free:
                    available_gf_anywhere.add(s)
                else:
                    available_reg_anywhere.add(s)

        # 9. Calculate sandwiches needing to be made (total count)
        num_to_make_total_needed = max(0, num_unserved - len(suitable_sandwiches_anywhere))

        # 10. Count available 'notexist' objects
        available_sandwich_objects = sum(1 for fact in state if match(fact, "notexist", "*"))

        # 11. Calculate actual sandwiches that can be made (limited by objects)
        actual_can_make_objects = min(num_to_make_total_needed, available_sandwich_objects)

        # 12. Add make action cost
        h += actual_can_make_objects

        # 13. Add put_on_tray action cost
        # Sandwiches needing put_on_tray are those in the kitchen + those that will be made.
        num_put_on_tray_needed = len(suitable_sandwiches_kitchen) + actual_can_make_objects
        h += num_put_on_tray_needed

        # 14. Calculate sandwiches needing to be made (by type) for penalty check
        needed_gf = sum(1 for c in unserved_children if self.allergy_status.get(c, False))
        needed_regular = sum(1 for c in unserved_children if not self.allergy_status.get(c, True))

        gf_to_make = max(0, needed_gf - len(available_gf_anywhere))

        # Regular needs can use remaining available GF or available non-GF.
        # Count how many regular needs are NOT met by existing suitable sandwiches (GF or non-GF).
        # Total suitable sandwiches needed = num_unserved.
        # Total suitable sandwiches available = len(suitable_sandwiches_anywhere).
        # Sandwiches to make (total) = max(0, num_unserved - len(suitable_sandwiches_anywhere)).
        # We need to make a regular type sandwich if the total number of sandwiches to make
        # is greater than the number of GF sandwiches to make.
        reg_to_make = num_to_make_total_needed - gf_to_make


        needs_to_make_gf_type = gf_to_make > 0
        needs_to_make_reg_type = reg_to_make > 0

        # 15. Check ingredient availability (from state, assuming static)
        any_bread = any(match(fact, "at_kitchen_bread", "*") for fact in state)
        any_content = any(match(fact, "at_kitchen_content", "*") for fact in state)
        any_gf_bread = any(match(fact, "at_kitchen_bread", "*") and match(fact, "no_gluten_bread", get_parts(fact)[1]) for fact in state)
        any_gf_content = any(match(fact, "at_kitchen_content", "*") and match(fact, "no_gluten_content", get_parts(fact)[1]) for fact in state)


        # 16. Add penalty for unmakeable sandwiches
        penalty = 0
        if num_to_make_total_needed > 0:
            if available_sandwich_objects == 0:
                 # Cannot make any sandwich if no notexist object
                 penalty += num_to_make_total_needed * 1000 # Penalize each needed sandwich
            else:
                 # Check if we can make the types needed among those we *can* make (up to actual_can_make_objects)
                 can_make_needed_types = True
                 if needs_to_make_gf_type and (not any_gf_bread or not any_gf_content):
                      can_make_needed_types = False
                 # A regular sandwich can be made from GF ingredients, so check if *any* ingredients exist for regular.
                 if needs_to_make_reg_type and (not any_bread or not any_content):
                      can_make_needed_types = False

                 if not can_make_needed_types:
                      # Cannot make the required type(s) of sandwich among those needed
                      penalty += num_to_make_total_needed * 1000 # Penalize each needed sandwich

        h += penalty

        return h
