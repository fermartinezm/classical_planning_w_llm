from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
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
    # Ensure the number of parts matches the number of args in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children.
    It calculates the minimum required steps for each unserved child independently
    based on the availability and location of suitable sandwiches and ingredients,
    and sums these minimum costs.

    # Assumptions
    - Each unserved child needs one suitable sandwich.
    - A suitable sandwich is gluten-free for allergic children, and any sandwich otherwise.
    - The steps to serve a child can be broken down into stages: Make Sandwich (if needed) -> Put on Tray -> Move Tray -> Serve.
    - The heuristic assigns a cost based on the earliest stage required for each child.
    - Resource conflicts (multiple children needing the same sandwich/ingredient/tray) are ignored (additive heuristic).
    - A large penalty is applied if a child cannot be served from the current state (e.g., no ingredients/sandwiches/notexist objects available).

    # Heuristic Initialization
    - Extract static facts like child allergies and gluten-free ingredient types.

    # Step-By-Step Thinking for Computing Heuristic
    For each child that is waiting but not yet served:
    1.  Check if a suitable sandwich is already on a tray at the child's waiting place. If yes, cost for this child is 1 (serve action).
    2.  Else, check if a suitable sandwich is on a tray at any other place. If yes, cost for this child is 2 (move tray + serve actions).
    3.  Else, check if a suitable sandwich is available in the kitchen. If yes, cost for this child is 3 (put on tray + move tray + serve actions).
    4.  Else, check if ingredients and a 'notexist' sandwich object are available in the kitchen to make a suitable sandwich. If yes, cost for this child is 4 (make + put on tray + move tray + serve actions).
    5.  If none of the above conditions are met (e.g., no resources to make a sandwich), assign a high cost (e.g., 1000) for this child, indicating a likely dead end.

    The total heuristic value is the sum of the minimum costs calculated for each unserved child.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts:
        - Child allergy status.
        - Gluten-free ingredient types.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Map child name to allergy status (True if allergic, False otherwise)
        self.child_allergy_map = {}
        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                self.child_allergy_map[get_parts(fact)[1]] = True
            elif match(fact, "not_allergic_gluten", "*"):
                self.child_allergy_map[get_parts(fact)[1]] = False

        # Sets of gluten-free ingredient names (static)
        self.is_gf_bread = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")}
        self.is_gf_content = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.
        total_cost = 0  # Initialize action cost counter.

        # 1. Identify unserved children
        unserved_children = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                child = get_parts(goal)[1]
                if goal not in state: # Child is not yet served
                    unserved_children.add(child)

        if not unserved_children:
            return 0 # Goal reached

        # Extract relevant facts from the current state for quick lookup
        # Note: is_gf_sandwich is NOT static, it depends on which sandwiches have been made
        is_gf_sandwich = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        at_kitchen_bread_set = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        at_kitchen_content_set = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}
        notexist_sandwich_set = {get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")}
        at_kitchen_sandwich_set = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        # Map sandwich name to tray name for sandwiches currently on trays
        ontray_map = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "ontray", "*", "*")}
        # Map tray name to place name for trays currently at places
        at_map = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")}
        # Map child name to waiting place
        waiting_map = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "waiting", "*", "*")}

        # Calculate counts for "can make" check (optimistic: just checks existence of at least one)
        avail_gf_bread_count = len(at_kitchen_bread_set.intersection(self.is_gf_bread))
        avail_any_bread_count = len(at_kitchen_bread_set)
        avail_gf_content_count = len(at_kitchen_content_set.intersection(self.is_gf_content))
        avail_any_content_count = len(at_kitchen_content_set)
        avail_notexist_s_count = len(notexist_sandwich_set)

        # 2. Calculate cost for each unserved child independently
        for child in unserved_children:
            place = waiting_map.get(child)
            if place is None:
                 # Child is unserved but not waiting anywhere? This state might be unreachable
                 # or indicates a problem setup issue. Assign a high cost.
                 total_cost += 1000
                 continue

            is_allergic = self.child_allergy_map.get(child, False) # Default to not allergic if status not in static

            # Check 1: Suitable sandwich on a tray at the child's place? (Cost 1: Serve)
            found_at_place = False
            for s, t in ontray_map.items():
                if at_map.get(t) == place: # Check if the tray holding the sandwich is at the child's place
                    s_is_gf = s in is_gf_sandwich
                    if (is_allergic and s_is_gf) or (not is_allergic):
                        found_at_place = True
                        break # Found one suitable sandwich at the correct location
            if found_at_place:
                total_cost += 1
                continue # Move to the next child

            # Check 2: Suitable sandwich on a tray elsewhere? (Cost 2: Move Tray + Serve)
            found_on_tray_elsewhere = False
            for s, t in ontray_map.items():
                 current_tray_place = at_map.get(t)
                 # Check if the tray holding the sandwich is at *any* place, but not the child's place
                 if current_tray_place is not None and current_tray_place != place:
                      s_is_gf = s in is_gf_sandwich
                      if (is_allergic and s_is_gf) or (not is_allergic):
                          found_on_tray_elsewhere = True
                          break # Found one suitable sandwich on a tray elsewhere
            if found_on_tray_elsewhere:
                 total_cost += 2
                 continue # Move to the next child

            # Check 3: Suitable sandwich in the kitchen? (Cost 3: Put on Tray + Move Tray + Serve)
            found_in_kitchen = False
            for s in at_kitchen_sandwich_set:
                 s_is_gf = s in is_gf_sandwich
                 if (is_allergic and s_is_gf) or (not is_allergic):
                     found_in_kitchen = True
                     break # Found one suitable sandwich in the kitchen
            if found_in_kitchen:
                 total_cost += 3
                 continue # Move to the next child

            # Check 4: Can a suitable sandwich be made? (Cost 4: Make + Put on Tray + Move Tray + Serve)
            # This check is optimistic, assuming ingredients and a notexist object are available.
            can_make_suitable = False
            if is_allergic:
                 # To make a GF sandwich, need at least one GF bread, one GF content, and one notexist sandwich object
                 if avail_gf_bread_count > 0 and avail_gf_content_count > 0 and avail_notexist_s_count > 0:
                     can_make_suitable = True
            else:
                 # To make any sandwich, need at least one any bread, one any content, and one notexist sandwich object
                 if avail_any_bread_count > 0 and avail_any_content_count > 0 and avail_notexist_s_count > 0:
                     can_make_suitable = True

            if can_make_suitable:
                 total_cost += 4
                 continue # Move to the next child

            # If none of the above conditions are met for a child, it means:
            # - No suitable sandwich is at their location.
            # - No suitable sandwich is on a tray elsewhere.
            # - No suitable sandwich is in the kitchen.
            # - Ingredients and notexist S are *not* available to make a suitable sandwich.
            # This implies the child cannot be served from this state unless resources appear (which they don't).
            # This state is likely a dead end or requires actions not considered (like creating new objects).
            # Assign a very high cost to prune this branch in a greedy search.
            total_cost += 1000 # Assign a high penalty

        return total_cost
