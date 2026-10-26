from fnmatch import fnmatch
# Assuming heuristic_base.py is in a directory named 'heuristics' relative to where this file will be placed.
# If running as a standalone file for testing, you might need to mock or adjust this import.
# For the final output, I will use the specified import path.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Provide a mock Heuristic class for standalone testing if needed
    print("Warning: Could not import Heuristic base class. Using mock class.")
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
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
    This heuristic estimates the number of actions required to serve all waiting children.
    It calculates the minimum estimated cost for each unserved child independently,
    based on the current state of suitable sandwiches and trays, and sums these costs.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - A suitable sandwich is gluten-free for allergic children and any sandwich for non-allergic children.
    - The actions required to serve a child are: make sandwich (if needed), put on tray, move tray to child's location, and serve.
    - Resources (ingredients, sandwich objects, trays) are assumed to be available when needed for the heuristic calculation (i.e., we check if *a* suitable item/ingredient exists to enable the next step for *a* child, not if there are enough for all simultaneous needs).
    - The cost of each action is 1.

    # Heuristic Initialization
    - Extracts static information about which children are allergic and which ingredients are gluten-free.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all unserved children and their waiting places from the current state.
    2. For each unserved child, determine if they are allergic using static information.
    3. For each unserved child, estimate the minimum number of actions required to serve them, based on the current state of suitable sandwiches, prioritizing the most advanced stage:
       - If the child is already served (goal predicate is true), the cost contribution for this child is 0.
       - Otherwise, the child needs a suitable sandwich delivered and served.
       - Check if there is a suitable sandwich already on a tray that is currently located at the child's waiting place. If yes, the estimated minimum cost for this child is 1 (only the 'serve' action is needed).
       - If not, check if there is a suitable sandwich on a tray that is currently located at a *different* place. If yes, the estimated minimum cost is 2 (a 'move_tray' action to bring the tray to the child's location, followed by the 'serve' action).
       - If not, check if there is a suitable sandwich currently in the kitchen (`at_kitchen_sandwich`). If yes, the estimated minimum cost is 3 (a 'put_on_tray' action, followed by a 'move_tray' action to the child's location, followed by the 'serve' action). This assumes a tray is available in the kitchen.
       - If not, check if the ingredients and an available sandwich object exist in the kitchen to 'make' a suitable sandwich. If yes, the estimated minimum cost is 4 (a 'make_sandwich' action, followed by 'put_on_tray', 'move_tray', and 'serve' actions). This assumes a tray is available in the kitchen.
       - If none of the above conditions are met (i.e., no suitable sandwich exists in any stage, and one cannot be made with available resources), the state is likely unsolvable from this point, and the heuristic returns infinity.
    4. The total heuristic value is the sum of these minimum estimated costs for each unserved child.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        # task.goals is available but not directly used in this heuristic's calculation logic
        # task.static contains facts that are true in all states
        static_facts = task.static

        # Extract static information
        self.allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")
        }
        self.no_gluten_bread = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")
        }
        self.no_gluten_content = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")
        }
        # We don't strictly need the set of all children or non-allergic children
        # as we iterate through waiting children in __call__ and check if they are allergic.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Parse state facts into useful data structures
        served_children = set() # {child_name}
        waiting_children = {} # {child_name: place_name}
        sandwiches_kitchen = set() # {sandwich_name}
        sandwiches_ontray = {} # {sandwich_name: tray_name}
        tray_locations = {} # {tray_name: place_name}
        sandwiches_gluten_free_state = set() # {sandwich_name}
        sandwich_objects_notexist = set() # {sandwich_name}
        bread_kitchen = set() # {bread_name}
        content_kitchen = set() # {content_name}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip potential malformed facts

            predicate = parts[0]
            if predicate == "served":
                if len(parts) > 1:
                    served_children.add(parts[1])
            elif predicate == "waiting":
                if len(parts) > 2: # Ensure fact has enough parts
                    child, place = parts[1], parts[2]
                    waiting_children[child] = place
            elif predicate == "at_kitchen_sandwich":
                 if len(parts) > 1:
                    sandwiches_kitchen.add(parts[1])
            elif predicate == "ontray":
                 if len(parts) > 2:
                    sandwich, tray = parts[1], parts[2]
                    sandwiches_ontray[sandwich] = tray
            elif predicate == "at":
                 # Assuming 'at' facts always involve a tray and a place based on domain
                 if len(parts) > 2:
                    obj, place = parts[1], parts[2]
                    # The domain says `(at ?t - tray ?p - place)`. So the first arg is always a tray.
                    tray_locations[obj] = place # Store location for any object that has 'at' predicate

            elif predicate == "no_gluten_sandwich":
                 if len(parts) > 1:
                    sandwiches_gluten_free_state.add(parts[1])
            elif predicate == "notexist":
                 if len(parts) > 1:
                    sandwich = parts[1]
                    sandwich_objects_notexist.add(sandwich)
            elif predicate == "at_kitchen_bread":
                 if len(parts) > 1:
                    bread_kitchen.add(parts[1])
            elif predicate == "at_kitchen_content":
                 if len(parts) > 1:
                    content_kitchen.add(parts[1])

        total_heuristic_cost = 0

        # Iterate through all children who are waiting but not yet served
        unserved_waiting_children = {
            child: place for child, place in waiting_children.items()
            if child not in served_children
        }

        # If no children are waiting or unserved, the goal is reached.
        if not unserved_waiting_children:
             return 0

        for child, place in unserved_waiting_children.items():
            is_allergic = child in self.allergic_children

            # Determine if a suitable sandwich exists in various stages, prioritizing closer stages
            min_cost_for_child = float('inf') # Initialize with infinity

            # Check State 1: Suitable sandwich on a tray at the child's place
            found_state_1 = False
            for s, t in sandwiches_ontray.items():
                # Check if sandwich is suitable
                is_gf_sandwich = s in sandwiches_gluten_free_state
                # Suitability: Allergic needs GF. Non-allergic can eat any.
                is_suitable = (is_allergic and is_gf_sandwich) or (not is_allergic)

                if is_suitable:
                    # Check if the tray is at the child's place
                    if t in tray_locations and tray_locations[t] == place:
                        min_cost_for_child = 1 # Serve action
                        found_state_1 = True
                        break # Found the best case for this child

            if found_state_1:
                total_heuristic_cost += min_cost_for_child
                continue # Move to the next child

            # Check State 2: Suitable sandwich on a tray at a different place
            found_state_2 = False
            for s, t in sandwiches_ontray.items():
                 # Check if sandwich is suitable
                 is_gf_sandwich = s in sandwiches_gluten_free_state
                 is_suitable = (is_allergic and is_gf_sandwich) or (not is_allergic)

                 if is_suitable:
                     # Check if the tray is NOT at the child's place (but exists)
                     if t in tray_locations and tray_locations[t] != place:
                         min_cost_for_child = 2 # Move tray + Serve action
                         found_state_2 = True
                         break # Found the next best case for this child

            if found_state_2:
                total_heuristic_cost += min_cost_for_child
                continue # Move to the next child

            # Check State 3: Suitable sandwich in the kitchen
            found_state_3 = False
            for s in sandwiches_kitchen:
                 # Check if sandwich is suitable
                 is_gf_sandwich = s in sandwiches_gluten_free_state
                 is_suitable = (is_allergic and is_gf_sandwich) or (not is_allergic)

                 if is_suitable:
                     min_cost_for_child = 3 # Put on tray + Move tray + Serve action
                     found_state_3 = True
                     break # Found the next best case for this child

            if found_state_3:
                total_heuristic_cost += min_cost_for_child
                continue # Move to the next child

            # Check State 4: Suitable sandwich needs to be made
            # Determine if a suitable sandwich *can* be made based on available ingredients and objects.
            # Need at least one suitable bread, one suitable content, and one notexist sandwich object.
            can_make_gf = bool(self.no_gluten_bread.intersection(bread_kitchen)) and \
                          bool(self.no_gluten_content.intersection(content_kitchen)) and \
                          bool(sandwich_objects_notexist)

            # Non-allergic can use any bread/content
            can_make_reg = bool(bread_kitchen) and \
                           bool(content_kitchen) and \
                           bool(sandwich_objects_notexist)

            suitable_sandwich_can_be_made = False
            if is_allergic:
                suitable_sandwich_can_be_made = can_make_gf
            else: # Not allergic, can eat regular or GF
                suitable_sandwich_can_be_made = can_make_reg or can_make_gf

            if suitable_sandwich_can_be_made:
                 min_cost_for_child = 4 # Make + Put on tray + Move tray + Serve action
                 # No need for found_state_4 flag, this is the last check before infinity
            else:
                 # If we reach here, no suitable sandwich exists or can be made with available resources.
                 # This state is likely unsolvable.
                 min_cost_for_child = float('inf')


            # Add the minimum cost found for this child to the total heuristic cost
            # If min_cost_for_child is infinity for any child, the total will become infinity.
            total_heuristic_cost += min_cost_for_child


        # Return the total estimated cost to serve all unserved children.
        return total_heuristic_cost
