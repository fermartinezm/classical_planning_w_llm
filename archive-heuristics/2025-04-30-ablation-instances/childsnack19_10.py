from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children
    who are currently waiting and not yet served. It calculates the cost for
    each unserved child independently based on the availability and location
    of a suitable sandwich.

    # Assumptions
    - Each child requires exactly one sandwich to be served.
    - Sufficient bread and content portions are available in the kitchen to make any required sandwiches.
    - Trays are available as needed.
    - The heuristic does not account for resource contention (e.g., multiple children needing the same sandwich or tray simultaneously).
    - The goal is to serve all children who are initially waiting and are listed in the goal conditions as needing to be served.

    # Heuristic Initialization
    - Identify all children who are initially waiting (from static facts).
    - Identify all children who are required to be served in the goal conditions.
    - The set of children to serve is the intersection of these two sets.
    - For each child in the set to serve, determine their waiting place and allergy status (gluten allergic or not) from the static facts. Store this information in dictionaries for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize the total heuristic cost to 0.
    2. Iterate through each child that was identified as needing to be served during initialization.
    3. For the current child, check if the predicate `(served <child_name>)` is present in the current state. If it is, this child is already served, and the cost for this child is 0; proceed to the next child.
    4. If the child is not served, determine the type of sandwich required: check the child's allergy status stored during initialization. If the child is allergic to gluten, a gluten-free sandwich is required; otherwise, any sandwich is suitable.
    5. Initialize the estimated cost for this unserved child to 4. This represents the maximum number of major steps potentially needed: (1) make the sandwich, (2) put it on a tray, (3) move the tray to the child's location, and (4) serve the child.
    6. Scan the current state to determine the availability and location of suitable sandwiches:
        - `found_suitable_sandwich`: A boolean flag, initially False. Set to True if *any* sandwich `S` is found in the state (either `at_kitchen_sandwich S` or `ontray S T`) that is suitable for the current child (gluten-free if required, checked against `no_gluten_sandwich` facts in the state).
        - `found_suitable_sandwich_on_tray`: A boolean flag, initially False. Set to True if *any* suitable sandwich `S` is found that is `ontray S T` for some tray `T`.
        - `found_suitable_sandwich_on_tray_at_place`: A boolean flag, initially False. Set to True if *any* suitable sandwich `S` is found that is `ontray S T` for some tray `T`, and that tray `T` is `at <child_place>` where `<child_place>` is the waiting location of the current child.
    7. Iterate through all facts in the current state to update the flags from step 6.
        - For facts matching `(at_kitchen_sandwich ?s)`: Extract `?s`. Check if `?s` is suitable (considering allergy and `no_gluten_sandwich` facts in the state). If yes, set `found_suitable_sandwich = True`.
        - For facts matching `(ontray ?s ?t)`: Extract `?s`, `?t`. Check if `?s` is suitable. If yes, set `found_suitable_sandwich = True` and `found_suitable_sandwich_on_tray = True`. Then, check if `(at ?t <child_place>)` is also in the state. If yes, set `found_suitable_sandwich_on_tray_at_place = True`.
        - Note: The check for `no_gluten_sandwich` must also be done against the current state facts.
    8. Adjust the child's estimated cost based on the flags:
        - If `found_suitable_sandwich` is True, decrement the child's cost by 1 (the 'make' action is saved).
        - If `found_suitable_sandwich_on_tray` is True, decrement the child's cost by another 1 (the 'put on tray' action is saved).
        - If `found_suitable_sandwich_on_tray_at_place` is True, decrement the child's cost by another 1 (the 'move tray' action is saved).
    9. Add the final estimated cost for this child to the total heuristic cost.
    10. After iterating through all children, return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting information about children
        who need serving, their waiting places, and allergy statuses from
        the static facts and goal conditions.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Identify children who are initially waiting
        children_initially_waiting = set()
        child_place_temp = {}
        child_allergy_temp = {} # True if allergic, False otherwise

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == "waiting" and len(parts) == 3:
                child, place = parts[1], parts[2]
                children_initially_waiting.add(child)
                child_place_temp[child] = place
            elif predicate == "allergic_gluten" and len(parts) == 2:
                child = parts[1]
                child_allergy_temp[child] = True
            elif predicate == "not_allergic_gluten" and len(parts) == 2:
                 child = parts[1]
                 child_allergy_temp[child] = False

        # 2. Identify children who are required to be served in the goal
        children_in_goal = set()
        for goal_fact in self.goals:
             parts = get_parts(goal_fact)
             if parts and parts[0] == "served" and len(parts) == 2:
                 children_in_goal.add(parts[1])

        # 3. The children to serve are those initially waiting AND in the goal
        self.children_to_serve = children_initially_waiting.intersection(children_in_goal)

        # Store place and allergy info only for the children we need to serve
        self.child_place = {c: child_place_temp[c] for c in self.children_to_serve if c in child_place_temp}
        self.child_allergy = {c: child_allergy_temp.get(c, False) for c in self.children_to_serve} # Default to False if allergy missing

        # Basic validation
        for child in list(self.children_to_serve):
             if child not in self.child_place:
                  print(f"Error: Child {child} needs serving (in goal) but has no waiting place defined in static facts. Removing from children_to_serve.")
                  self.children_to_serve.remove(child)
             # Allergy status defaults to False if missing, which is handled by .get()

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.
        total_cost = 0

        # Pre-scan the state to quickly build lookup structures
        existing_sandwiches = set()
        gluten_free_sandwiches_in_state = set() # Sandwiches that are currently marked as no_gluten_sandwich
        sandwich_on_tray = {} # Map sandwich -> tray
        tray_location = {} # Map tray -> place

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == "at_kitchen_sandwich" and len(parts) == 2:
                sandwich = parts[1]
                existing_sandwiches.add(sandwich)
            elif predicate == "ontray" and len(parts) == 3:
                sandwich, tray = parts[1], parts[2]
                existing_sandwiches.add(sandwich)
                sandwich_on_tray[sandwich] = tray
            elif predicate == "no_gluten_sandwich" and len(parts) == 2:
                sandwich = parts[1]
                gluten_free_sandwiches_in_state.add(sandwich)
            elif predicate == "at" and len(parts) == 3:
                 obj, place = parts[1], parts[2]
                 # Assume anything starting with 'tray' is a tray based on domain structure
                 if obj.startswith('tray'):
                     tray_location[obj] = place

        for child in self.children_to_serve:
            # Check if child is already served
            if f"(served {child})" in state:
                continue # Child is served, cost is 0 for this child

            child_place = self.child_place[child]
            is_allergic = self.child_allergy.get(child, False) # Should exist due to init cleanup, but safe

            # Base cost: make, put, move, serve
            cost_for_child = 4

            # Check for suitable sandwiches
            suitable_sandwiches_exist = False
            suitable_sandwich_on_tray_exists = False
            suitable_sandwich_on_tray_at_place_exists = False

            # Iterate through all sandwiches we know exist in the state (kitchen or on tray)
            for sandwich in existing_sandwiches:
                # Check if the sandwich is suitable for the child
                is_suitable = True
                if is_allergic and sandwich not in gluten_free_sandwiches_in_state:
                    is_suitable = False

                if is_suitable:
                    suitable_sandwiches_exist = True

                    # Check if the suitable sandwich is on a tray
                    if sandwich in sandwich_on_tray:
                        suitable_sandwich_on_tray_exists = True
                        tray = sandwich_on_tray[sandwich]

                        # Check if the tray is at the child's place
                        if tray in tray_location and tray_location[tray] == child_place:
                             suitable_sandwich_on_tray_at_place_exists = True
                             # Found the best case for this child, no need to check other sandwiches
                             break # Exit the inner loop over existing_sandwiches

            # Adjust cost based on findings
            if suitable_sandwiches_exist:
                cost_for_child -= 1 # Saved 'make' action

            if suitable_sandwich_on_tray_exists:
                cost_for_child -= 1 # Saved 'put_on_tray' action

            if suitable_sandwich_on_tray_at_place_exists:
                cost_for_child -= 1 # Saved 'move_tray' action

            total_cost += cost_for_child

        return total_cost
