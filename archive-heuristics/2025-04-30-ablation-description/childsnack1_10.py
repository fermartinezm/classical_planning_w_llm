from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, though planner typically provides strings
        return []
    return fact[1:-1].split()

# Helper function to match a PDDL fact string against a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
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
    This heuristic estimates the cost to serve all waiting children by summing up
    an estimated cost for each individual unserved child. The cost for a child
    depends on how "close" a suitable sandwich is to being served to them.

    # Assumptions
    - The goal is to serve all children who are specified in the task's goal predicates.
    - Each child requires exactly one sandwich.
    - Allergic children require gluten-free sandwiches.
    - Non-allergic children can eat any sandwich (regular or gluten-free).
    - Ingredients in the kitchen are sufficient to make any required sandwich type if needed (simplified assumption: only checks for existence of *any* suitable ingredients, not quantity).
    - The costs assigned to different stages (serve, move tray, put on tray, make sandwich) represent the minimum number of actions to reach that stage for a single child, ignoring resource contention (like multiple children needing the same tray or the last ingredient).

    # Heuristic Initialization
    - Extracts static information about which children are allergic from the static facts.
    - Identifies the set of children who are the targets of the 'served' goal predicates.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a state is the sum of the estimated costs for each child
    who is currently a target child (appears in the goal) but is not yet marked as `served` in the current state.

    For each target child `c`:

    1.  Check if the child `c` is already `served` in the current state. If yes, the cost for this child is 0.
    2.  If the child is not served, determine if the child is allergic to gluten based on static facts.
    3.  Find the child's current waiting place `p` from the state facts `(waiting c p)`. If the child is a target but not waiting, assign a high cost (indicating an unexpected or difficult state).
    4.  Determine the minimum estimated actions needed to serve this child, based on the current state:
        *   **Cost 1 (Serve):** Check if a suitable sandwich `s` is already on a tray `t` that is currently `at` the child's waiting `place p`. A sandwich `s` is suitable if it is gluten-free (`(no_gluten_sandwich s)` is in state) when the child `c` is allergic, or any sandwich otherwise. If such a sandwich exists, the cost for this child is 1.
        *   **Cost 2 (Move Tray + Serve):** If Cost 1 is not met, check if a suitable sandwich `s` is on *any* tray `t` (regardless of tray location). If such a sandwich exists, the cost for this child is 2 (estimated cost of moving the tray + serving).
        *   **Cost 3 (Put on Tray + Move Tray + Serve):** If Costs 1 and 2 are not met, check if a suitable sandwich `s` is available `at_kitchen_sandwich`. If such a sandwich exists, the cost for this child is 3 (estimated cost of putting on tray + moving tray + serving).
        *   **Cost 4 (Make + Put on Tray + Move Tray + Serve):** If Costs 1, 2, and 3 are not met (no suitable sandwich exists anywhere), check if suitable ingredients are available in the kitchen to make one. Suitable ingredients mean at least one available gluten-free bread and one available gluten-free content if the child is allergic, or at least one available bread and one available content otherwise. If suitable ingredients are available, the cost for this child is 4 (estimated cost of making sandwich + putting on tray + moving tray + serving).
        *   **Cost 1000 (Unsolvable/High Cost):** If none of the above conditions are met (i.e., no suitable sandwich exists and suitable ingredients are *not* available to make one). This indicates the state might be unsolvable for this child with the current resources. A large cost (1000) is assigned to penalize such states heavily.

    5.  Sum the minimum estimated cost for each unserved target child to get the total heuristic value for the state.

    The heuristic is 0 if and only if all children specified in the goal are marked as `served`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals  # The set of goal facts (e.g., served children)
        self.static = task.static # Static facts (e.g., allergies)

        # Extract which children are allergic from static facts
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "allergic_gluten", "*")
        }

        # Identify the set of children who are the targets of the 'served' goal predicates.
        self.target_children = {
            get_parts(goal)[1]
            for goal in self.goals
            if match(goal, "served", "*")
        }


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # If the goal is reached, the heuristic is 0.
        if self.goals <= state:
             return 0

        total_cost = 0

        # Collect available ingredients and sandwiches in the current state for quick lookup
        available_bread = {get_parts(f)[1] for f in state if match(f, "at_kitchen_bread", "*")}
        available_content = {get_parts(f)[1] for f in state if match(f, "at_kitchen_content", "*")}
        available_gf_bread = {b for b in available_bread if '(no_gluten_bread ' + b + ')' in state}
        available_gf_content = {c for c in available_content if '(no_gluten_content ' + c + ')' in state}

        sandwiches_in_kitchen = {get_parts(f)[1] for f in state if match(f, "at_kitchen_sandwich", "*")}
        sandwiches_on_trays = {} # Map sandwich -> tray
        for fact in state:
             if match(fact, "ontray", "*", "*"):
                 s, t = get_parts(fact)[1:]
                 sandwiches_on_trays[s] = t

        tray_locations = {} # Map tray -> place
        for fact in state:
             if match(fact, "at", "*", "*"):
                 # Check if the object is a tray (assuming 'at' predicate is only for trays and robot)
                 # In childsnacks, 'at' is only for trays.
                 parts = get_parts(fact)
                 if len(parts) == 3: # Should be (at ?tray ?place)
                     tray, place = parts[1:]
                     tray_locations[tray] = place


        # Iterate through the children we need to serve (those in the goal)
        for child in self.target_children:
            # Check if the child is already served
            if '(served ' + child + ')' in state:
                continue # This child is served, cost is 0 for them

            # Child is not served. Estimate cost for this child.
            child_cost = 0
            child_is_allergic = child in self.allergic_children

            # Find the child's current waiting place from the state
            waiting_place = None
            for fact in state:
                 if match(fact, "waiting", child, "*"):
                     waiting_place = get_parts(fact)[2]
                     break

            if waiting_place is None:
                 # This child is a target child but not currently waiting.
                 # Assign a high cost as this state is unexpected or potentially unsolvable
                 # within the standard action sequence for this child.
                 total_cost += 1000
                 continue

            # --- Check for Cost 1: Suitable sandwich on a tray at the child's location ---
            found_cost_1 = False
            for s, t in sandwiches_on_trays.items():
                if t in tray_locations and tray_locations[t] == waiting_place:
                    # Check if the sandwich is suitable for the child's allergy status
                    is_suitable = (not child_is_allergic) or ('(no_gluten_sandwich ' + s + ')' in state)
                    if is_suitable:
                        child_cost = 1
                        found_cost_1 = True
                        break # Found the best case for this child

            if found_cost_1:
                total_cost += child_cost
                continue # Move to the next child

            # --- Check for Cost 2: Suitable sandwich on any tray ---
            found_cost_2 = False
            for s in sandwiches_on_trays.keys():
                 # Check if the sandwich is suitable for the child's allergy status
                 is_suitable = (not child_is_allergic) or ('(no_gluten_sandwich ' + s + ')' in state)
                 if is_suitable:
                     child_cost = 2
                     found_cost_2 = True
                     break # Found the next best case

            if found_cost_2:
                total_cost += child_cost
                continue # Move to the next child

            # --- Check for Cost 3: Suitable sandwich in the kitchen ---
            found_cost_3 = False
            for s in sandwiches_in_kitchen:
                 # Check if the sandwich is suitable for the child's allergy status
                 is_suitable = (not child_is_allergic) or ('(no_gluten_sandwich ' + s + ')' in state)
                 if is_suitable:
                     child_cost = 3
                     found_cost_3 = True
                     break # Found the next best case

            if found_cost_3:
                total_cost += child_cost
                continue # Move to the next child

            # --- Check for Cost 4: Suitable ingredients available to make a sandwich ---
            suitable_ingredients_available = False
            if child_is_allergic:
                # Need at least one GF bread and one GF content in the kitchen
                if len(available_gf_bread) > 0 and len(available_gf_content) > 0:
                    suitable_ingredients_available = True
            else: # not allergic
                # Need at least one any bread and one any content in the kitchen
                if len(available_bread) > 0 and len(available_content) > 0:
                    suitable_ingredients_available = True

            if suitable_ingredients_available:
                child_cost = 4
            else:
                # Cannot make a suitable sandwich with available ingredients.
                # Assign a high cost indicating potential unsolvability or a very long path.
                child_cost = 1000

            total_cost += child_cost

        return total_cost
