from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children.
    It counts the number of sandwiches of each type (gluten-free or regular) needed
    and estimates the cost based on the current "stage" of the required sandwiches
    (ready at child's place, on tray elsewhere, in kitchen, or needing to be made).
    It prioritizes using sandwiches that are closer to being served, assigning a cost
    based on the minimum number of actions to get a sandwich from that stage to being served.

    # Assumptions
    - Each unserved child requires exactly one sandwich of the correct type (gluten-free for allergic, regular otherwise).
    - The base costs for actions are: serve=1, move_tray=1, put_on_tray=1, make_sandwich=1.
    - A sandwich needs to go through stages: (make) -> at_kitchen_sandwich -> ontray (at kitchen) -> move tray -> ontray (at place) -> served.
    - The heuristic assigns costs based on these stages:
        - Stage 4 (Ready at place): 1 action (serve)
        - Stage 3 (Ready on tray elsewhere/kitchen): 2 actions (move tray + serve)
        - Stage 2 (Ready in kitchen): 3 actions (put on tray + move tray + serve)
        - Stage 1 (Needs making): 4 actions (make + put on tray + move tray + serve)
    - The heuristic simplifies tray logistics by assuming sufficient trays are available when needed at the kitchen for putting sandwiches on, and focuses on the cost per sandwich based on its current stage.
    - Unsolvable states (e.g., not enough ingredients or sandwich slots) result in an infinite heuristic value.

    # Heuristic Initialization
    - Extracts static information about children's allergies and waiting places.
    - Identifies which bread and content types are gluten-free based on static facts.
    - Collects the names of all children from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who have not yet been served by checking the state against the list of all children from static facts. Categorize these unserved children by their allergy status (gluten-free needed vs. regular needed). Count the total number of GF and Regular sandwiches required (`needs_gf`, `needs_reg`).
    2. If no children are unserved, the goal is reached, return 0.
    3. Identify all unique places where unserved children are waiting (`unserved_places`).
    4. Count available ingredients (bread, content) and sandwich slots (`notexist`) in the kitchen from the current state. Distinguish between gluten-free and regular ingredients based on static facts.
    5. Count existing sandwiches in the current state and categorize them by type (GF/Regular, based on `no_gluten_sandwich` predicate) and their current stage:
       - Stage 4 (Ready at place): Sandwich is `ontray` on a tray located at a place in `unserved_places`.
       - Stage 3 (Ready on tray elsewhere): Sandwich is `ontray` on a tray located at kitchen or a place *not* in `unserved_places`.
       - Stage 2 (Ready in kitchen): Sandwich is `at_kitchen_sandwich`.
    6. Calculate the maximum number of GF and Regular sandwiches that *can* be made based on available `notexist` slots and ingredients (`can_make_gf`, `can_make_reg`). Note that `notexist` slots are a shared resource.
    7. Calculate the total heuristic cost by greedily satisfying the sandwich needs (`needs_gf`, `needs_reg`) from the cheapest available stages first:
       - For GF needs, use available GF sandwiches from Stage 4 (cost 1), then Stage 3 (cost 2), then Stage 2 (cost 3), then make new ones (Stage 1, cost 4).
       - When using Stage 1 (making sandwiches), decrement the count of available `notexist` slots.
       - Repeat the greedy satisfaction process for Regular needs, using available Regular sandwiches from Stages 4, 3, 2, and then making new ones (Stage 1). The `can_make_reg` calculation must use the *remaining* `available_notexist` slots after satisfying GF needs.
    8. If, after exhausting all available sandwiches and make capacity, there are still unserved children (`needs_gf > 0` or `needs_reg > 0`), the state is likely unsolvable, return infinity.
    9. Otherwise, return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        super().__init__(task) # Call the base class constructor

        self.child_allergy = {}
        self.child_place = {}
        self.gf_bread_types = set()
        self.gf_content_types = set()
        self.all_children = set()

        # Extract static facts
        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "allergic_gluten":
                child = parts[1]
                self.child_allergy[child] = 'gluten'
                self.all_children.add(child)
            elif predicate == "not_allergic_gluten":
                child = parts[1]
                self.child_allergy[child] = 'none'
                self.all_children.add(child)
            elif predicate == "waiting":
                child, place = parts[1], parts[2]
                self.child_place[child] = place
            elif predicate == "no_gluten_bread":
                bread = parts[1]
                self.gf_bread_types.add(bread)
            elif predicate == "no_gluten_content":
                content = parts[1]
                self.gf_content_types.add(content)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state

        # 1. Identify unserved children and their needs
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = {c for c in self.all_children if c not in served_children}

        if not unserved_children:
            return 0 # Goal state reached

        unserved_places = {self.child_place.get(c) for c in unserved_children if c in self.child_place} # Get places for unserved children

        needs_gf = sum(1 for c in unserved_children if self.child_allergy.get(c) == 'gluten')
        needs_reg = sum(1 for c in unserved_children if self.child_allergy.get(c) == 'none')

        # 2. Count available ingredients and slots
        available_notexist = sum(1 for fact in state if match(fact, "notexist", "*"))
        available_gf_bread = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and get_parts(fact)[1] in self.gf_bread_types)
        available_reg_bread = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and get_parts(fact)[1] not in self.gf_bread_types)
        available_gf_content = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and get_parts(fact)[1] in self.gf_content_types)
        available_reg_content = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and get_parts(fact)[1] not in self.gf_content_types)

        # 3. Count existing sandwiches by type and stage
        kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        ontray_sandwiches = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "ontray", "*", "*")} # Map sandwich to tray
        tray_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith('tray')} # Map tray to location
        gf_sandwiches_state = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")} # Set of GF sandwiches that exist

        ready_gf_at_place = 0
        ready_reg_at_place = 0
        ready_gf_elsewhere = 0
        ready_reg_elsewhere = 0
        ready_gf_kitchen = 0
        ready_reg_kitchen = 0

        # Count kitchen sandwiches
        for s in kitchen_sandwiches:
            if s in gf_sandwiches_state:
                ready_gf_kitchen += 1
            else:
                ready_reg_kitchen += 1

        # Count on-tray sandwiches
        for s, t in ontray_sandwiches.items():
            is_gf = s in gf_sandwiches_state
            p = tray_locations.get(t) # Get tray location

            if p is not None: # Tray must have a location
                if p in unserved_places:
                    if is_gf:
                        ready_gf_at_place += 1
                    else:
                        ready_reg_at_place += 1
                else: # Tray is at kitchen or a place with no unserved children
                     if is_gf:
                        ready_gf_elsewhere += 1
                     else:
                        ready_reg_elsewhere += 1
            # else: This case implies an invalid state where a sandwich is on a tray
            # but the tray has no location. We'll ignore such sandwiches for the heuristic.


        # 4. Calculate maximum makeable sandwiches
        can_make_gf = min(available_notexist, available_gf_bread, available_gf_content)
        can_make_reg_initial = min(available_notexist, available_reg_bread, available_reg_content)


        # 5. Perform greedy cost calculation
        total_cost = 0

        # Satisfy GF needs
        use = min(needs_gf, ready_gf_at_place)
        total_cost += use * 1
        needs_gf -= use

        use = min(needs_gf, ready_gf_elsewhere)
        total_cost += use * 2
        needs_gf -= use

        use = min(needs_gf, ready_gf_kitchen)
        total_cost += use * 3
        needs_gf -= use

        # Recalculate can_make_gf based on remaining notexist slots
        # Note: This recalculation is only needed if we consumed notexist slots in previous stages,
        # which we don't in this cost model (1,2,3 don't use notexist).
        # But making sandwiches does consume notexist.
        # Let's just use the initial can_make_gf here and update notexist.
        use = min(needs_gf, can_make_gf)
        total_cost += use * 4
        needs_gf -= use
        # Consume resources for made sandwiches (only notexist matters for the other type's can_make)
        available_notexist -= use
        # available_gf_bread -= use # Not needed for calculating can_make_reg
        # available_gf_content -= use # Not needed for calculating can_make_reg


        if needs_gf > 0:
            return float('inf') # Cannot satisfy all GF needs

        # Recalculate can_make_reg based on remaining notexist slots
        can_make_reg = min(available_notexist, available_reg_bread, available_reg_content)

        # Satisfy Regular needs
        use = min(needs_reg, ready_reg_at_place)
        total_cost += use * 1
        needs_reg -= use

        use = min(needs_reg, ready_reg_elsewhere)
        total_cost += use * 2
        needs_reg -= use

        use = min(needs_reg, ready_reg_kitchen)
        total_cost += use * 3
        needs_reg -= use

        use = min(needs_reg, can_make_reg)
        total_cost += use * 4
        needs_reg -= use
        # Consume resources for made sandwiches
        # available_notexist -= use # Already updated after GF
        # available_reg_bread -= use # Not needed
        # available_reg_content -= use # Not needed

        if needs_reg > 0:
            return float('inf') # Cannot satisfy all Regular needs

        return total_cost
