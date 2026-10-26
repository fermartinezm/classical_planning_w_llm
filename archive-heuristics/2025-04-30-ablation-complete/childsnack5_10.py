from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided elsewhere
# In a real scenario, this would be imported from the planner's framework
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.strip() or fact[0] != '(' or fact[-1] != ')':
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
    It sums up the estimated costs for:
    1. Serving each unserved child (representing the final 'serve' action).
    2. Making sandwiches that are needed but not yet made.
    3. Putting sandwiches onto trays that need to be on trays but aren't yet.
    4. Moving trays to locations where children need sandwiches of a type not currently available on a suitable tray at that location.

    # Assumptions
    - Resources (bread, content, sandwich objects) are sufficient to make needed sandwiches eventually.
    - Trays are available to put sandwiches on.
    - Any tray can be moved to any place.
    - The heuristic does not track specific objects (which sandwich goes to which child, which tray is used). It counts required actions based on types and locations.
    - The heuristic is non-admissible, designed for greedy best-first search to minimize node expansions.

    # Heuristic Initialization
    The heuristic pre-processes static facts from the task definition:
    - Identifies which children are allergic to gluten and which are not.
    - Records the waiting place for each child.
    - Collects the set of all children in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1.  **Identify Unserved Children:** Determine the set and count of children who are waiting (based on static facts) but are not yet marked as 'served' in the current state. This count is a baseline estimate for the number of 'serve' actions needed.
    2.  **Count Sandwich Needs:** Based on the unserved children and their allergy status (from static facts), determine the total number of gluten-free and regular sandwiches required across all unserved children.
    3.  **Count Sandwiches Already Made:** Identify sandwiches in the current state that are either 'at_kitchen_sandwich' or 'ontray'. Separate these into gluten-free (using 'no_gluten_sandwich' predicate) and regular sandwiches.
    4.  **Count Sandwiches to Make:** Compare the total required sandwiches of each type (GF/Regular) with the number of sandwiches of that type that have already been made. The difference, if positive, represents the minimum number of 'make_sandwich' actions needed for each type.
    5.  **Count Sandwiches Already on Trays:** Identify sandwiches in the current state that are 'ontray'. Separate these into gluten-free and regular based on 'no_gluten_sandwich'.
    6.  **Count Sandwiches to Put on Trays:** Compare the total required sandwiches of each type with the number of sandwiches of that type that are already on trays. The difference, if positive, represents the minimum number of 'put_on_tray' actions needed for each type (assuming the sandwiches are made or will be made).
    7.  **Identify Tray Delivery Needs:** For each place where unserved children are waiting (grouped by allergy type), check if there is already a tray at that location (using 'at ?t ?p' predicate) containing a sandwich of the required type (GF for allergic children, Regular for non-allergic).
        - If unserved allergic children are at a place *and* no tray currently at that place has a GF sandwich on it, count one 'GF tray delivery' needed for that place.
        - If unserved non-allergic children are at a place *and* no tray currently at that place has a Regular sandwich on it, count one 'Regular tray delivery' needed for that place.
    8.  **Count Tray Movements:** The total number of 'GF tray delivery' needs plus the total number of 'Regular tray delivery' needs across all places is an estimate of the minimum number of 'move_tray' actions required to bring necessary sandwiches to the children's locations.
    9.  **Sum Costs:** The total heuristic value is the sum of the counts from steps 1, 4, 6, and 8.

    This heuristic provides an estimate by summing up distinct types of actions or action enablers needed to progress towards the goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children's
        allergies and waiting places.
        """
        super().__init__(task)

        # Map child name to True (allergic) or False (not allergic)
        self.child_allergy = {}
        # Map child name to their waiting place
        self.child_place = {}
        # Set of all children
        self.all_children = set()

        # Extract static facts from the task
        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "allergic_gluten":
                child = parts[1]
                self.child_allergy[child] = True
                self.all_children.add(child)
            elif predicate == "not_allergic_gluten":
                child = parts[1]
                self.child_allergy[child] = False
                self.all_children.add(child)
            elif predicate == "waiting":
                child, place = parts[1], parts[2]
                self.child_place[child] = place
                # Ensure child is added even if allergy wasn't listed first
                self.all_children.add(child)

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # 1. Identify Unserved Children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = {c for c in self.all_children if c not in served_children}
        num_unserved_children = len(unserved_children)

        # If no children are unserved, the goal is reached.
        if num_unserved_children == 0:
            return 0

        # 2. Count Sandwich Needs based on unserved children
        num_gf_needed_total = sum(1 for c in unserved_children if self.child_allergy.get(c, False)) # Default to False if allergy not specified
        num_reg_needed_total = sum(1 for c in unserved_children if not self.child_allergy.get(c, True)) # Default to True if allergy not specified

        # 3. Count Sandwiches Already Made (at_kitchen_sandwich or ontray)
        gf_sandwiches_made_set = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        all_sandwiches_made_set = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")} | \
                                  {get_parts(fact)[1] for fact in state if match(fact, "ontray", "*", "*")}
        reg_sandwiches_made_set = all_sandwiches_made_set - gf_sandwiches_made_set

        num_gf_made = len(gf_sandwiches_made_set)
        num_reg_made = len(reg_sandwiches_made_set)

        # 4. Count Sandwiches to Make
        num_gf_to_make = max(0, num_gf_needed_total - num_gf_made)
        num_reg_to_make = max(0, num_reg_needed_total - num_reg_made)

        # 5. Count Sandwiches Already on Trays
        sandwiches_ontray_map = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "ontray", "*", "*")}
        gf_sandwiches_ontray_set = {s for s in sandwiches_ontray_map if s in gf_sandwiches_made_set}
        reg_sandwiches_ontray_set = {s for s in sandwiches_ontray_map if s in reg_sandwiches_made_set}

        num_gf_ontray_total = len(gf_sandwiches_ontray_set)
        num_reg_ontray_total = len(reg_sandwiches_ontray_set)

        # 6. Count Sandwiches to Put on Trays
        # These are sandwiches that need to end up on trays but aren't yet.
        num_gf_put_on_tray = max(0, num_gf_needed_total - num_gf_ontray_total)
        num_reg_put_on_tray = max(0, num_reg_needed_total - num_reg_ontray_total)

        # 7. Identify Tray Delivery Needs per Place
        tray_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("tray")}

        # Determine which trays have which type of sandwiches
        tray_contents = {t: {'gf': set(), 'reg': set()} for t in tray_locations}
        for s, t in sandwiches_ontray_map.items():
             if t in tray_contents: # Ensure tray exists and is located
                if s in gf_sandwiches_made_set:
                    tray_contents[t]['gf'].add(s)
                elif s in reg_sandwiches_made_set:
                    tray_contents[t]['reg'].add(s)

        # Identify places that currently have trays with GF/Regular sandwiches
        places_with_gf_tray = {tray_locations[t] for t, contents in tray_contents.items() if len(contents['gf']) > 0}
        places_with_reg_tray = {tray_locations[t] for t, contents in tray_contents.items() if len(contents['reg']) > 0}

        # Identify places where unserved children are waiting
        places_with_unserved_allergic = {self.child_place[c] for c in unserved_children if self.child_allergy.get(c, False)}
        places_with_unserved_non_allergic = {self.child_place[c] for c in unserved_children if not self.child_allergy.get(c, True)}

        # 8. Count Tray Movements (deliveries) needed
        # A delivery is needed for a place/type if children need that type there
        # and no suitable tray is currently at that place.
        num_gf_tray_deliveries = sum(1 for p in places_with_unserved_allergic if p not in places_with_gf_tray)
        num_reg_tray_deliveries = sum(1 for p in places_with_unserved_non_allergic if p not in places_with_reg_tray)

        # 9. Sum Costs
        # Base cost: 1 action per unserved child (the final serve action)
        heuristic_value = num_unserved_children

        # Add cost for making sandwiches that don't exist yet
        heuristic_value += num_gf_to_make
        heuristic_value += num_reg_to_make

        # Add cost for putting sandwiches onto trays if they need to be on trays
        # and aren't yet. This counts how many *more* sandwiches need to end up on trays.
        heuristic_value += num_gf_put_on_tray
        heuristic_value += num_reg_put_on_tray

        # Add cost for moving trays to places that need deliveries of a specific type
        # and don't have a suitable tray already there.
        heuristic_value += num_gf_tray_deliveries
        heuristic_value += num_reg_tray_deliveries

        return heuristic_value
