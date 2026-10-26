from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
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
    # Use zip to compare parts and args up to the length of the shorter sequence.
    # fnmatch handles the wildcard '*'.
    # This works correctly for the patterns used in this domain.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of unserved children, the number of sandwiches that
    still need to be made (considering allergy constraints and available sandwiches),
    the number of sandwiches currently in the kitchen that need to be put on a tray,
    and the number of distinct locations where unserved children are waiting but
    no tray is currently present.

    The heuristic is calculated as:
    h = (Number of unserved children)
      + (Number of distinct places needing a tray move)
      + (Number of sandwiches currently in the kitchen)
      + 2 * (Number of sandwiches that need to be made)

    Each unserved child needs a 'serve' action.
    Each distinct location without a tray needs a 'move_tray' action.
    Each sandwich in the kitchen needs a 'put_on_tray' action.
    Each sandwich that needs to be made requires a 'make_sandwich' action
    followed by a 'put_on_tray' action (hence the factor of 2).

    # Assumptions
    - Sufficient bread, content, and sandwich objects exist to make needed sandwiches.
    - Sufficient trays exist and are available at the kitchen initially or can be moved.
    - The cost of each action type (make, put, move, serve) is 1.
    - Non-allergic children can be served any type of sandwich.
    - Allergic children must be served gluten-free sandwiches.
    - When calculating sandwiches to make, we prioritize using available gluten-free
      sandwiches for allergic children, then use remaining available gluten-free
      and all available non-gluten-free sandwiches for non-allergic children.

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - Which children are allergic and non-allergic.
    - The waiting place for each child.
    - The set of all children in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children from static facts (those mentioned in allergic/not_allergic or waiting predicates).
    2. Determine which children are allergic and non-allergic from static facts.
    3. Determine the waiting place for each child from static facts.
    4. In the current state, identify which children are not yet served by checking for the absence of the `(served ?c)` fact. These are the unserved children.
    5. Separate unserved children into those who are allergic and those who are not, using the static information.
    6. Count the total number of unserved children (N_unserved). This contributes N_unserved to the heuristic (representing the final 'serve' action for each).
    7. Count the number of available sandwiches currently in the kitchen (`at_kitchen_sandwich`) or on any tray (`ontray`).
    8. Count how many of these available sandwiches are gluten-free (`no_gluten_sandwich`). Note: The `no_gluten_sandwich` predicate is a property of the sandwich *type*, not its location. We must check this property for sandwiches that are currently available (in kitchen or on tray).
    9. Calculate the number of new sandwiches that need to be made (`N_to_make`):
       - Calculate the number of gluten-free sandwiches required for allergic children that are not available (`max(0, N_allergic_unserved - avail_gf_sandwiches)`).
       - Calculate the number of *additional* sandwiches (can be any type) required for non-allergic children, after accounting for available non-GF sandwiches and any available GF sandwiches not needed by allergic children.
       - Sum these two counts to get `N_to_make`. This contributes 2 * N_to_make to the heuristic (representing the 'make' and subsequent 'put_on_tray' actions).
    10. Count the number of sandwiches currently located in the kitchen (`at_kitchen_sandwich`). These need a 'put_on_tray' action. This contributes N_kitchen_sandwiches to the heuristic.
    11. Identify all distinct places where unserved children are waiting (using static waiting facts and the list of unserved children).
    12. Identify all distinct places where trays are currently located (using `at ?t ?p` facts for trays).
    13. Count the number of places from step 11 that are not in step 12. These places need a tray moved there. This contributes N_places_need_tray to the heuristic (representing the 'move_tray' action).
    14. Sum the contributions from steps 6, 9 (multiplied by 2), 10, and 13 to get the total heuristic value.
    15. If N_unserved is 0, the goal is reached, and the heuristic is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = set()
        self.non_allergic_children = set()
        self.waiting_places = {} # Map child to place
        self.all_children = set()

        # Extract static information about children and their waiting places
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                child = parts[1]
                self.allergic_children.add(child)
                self.all_children.add(child)
            elif parts[0] == 'not_allergic_gluten':
                child = parts[1]
                self.non_allergic_children.add(child)
                self.all_children.add(child)
            elif parts[0] == 'waiting':
                child = parts[1]
                place = parts[2]
                self.waiting_places[child] = place
                # Add child to all_children set if not already added by allergy fact
                self.all_children.add(child)

        # Ensure all children mentioned in goals are also in self.all_children
        # (This handles cases where a child might only appear in the goal and waiting/allergy facts)
        for goal in self.goals:
             if match(goal, "served", "*"):
                 child = get_parts(goal)[1]
                 self.all_children.add(child)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 4. Identify unserved children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = self.all_children - served_children

        # 15. If all children are served, the goal is reached
        if not unserved_children:
            return 0

        # 5. Separate unserved children by allergy
        unserved_allergic = unserved_children.intersection(self.allergic_children)
        unserved_non_allergic = unserved_children.intersection(self.non_allergic_children)

        # 6. Count unserved children (for 'serve' action)
        N_unserved = len(unserved_children)

        # 7. Count available sandwiches (kitchen or tray)
        sandwiches_on_tray = set()
        sandwiches_in_kitchen = set()
        # 8. Identify GF sandwiches (those that exist and have the property)
        gf_sandwiches_property = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}


        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'ontray':
                sandwich = parts[1]
                sandwiches_on_tray.add(sandwich)
            elif parts[0] == 'at_kitchen_sandwich':
                sandwich = parts[1]
                sandwiches_in_kitchen.add(sandwich)

        # Count available sandwiches (those on tray or in kitchen)
        available_sandwiches_set = sandwiches_on_tray.union(sandwiches_in_kitchen)

        avail_gf_sandwiches = len(available_sandwiches_set.intersection(gf_sandwiches_property))
        avail_reg_sandwiches = len(available_sandwiches_set) # Total available sandwiches

        # 9. Calculate sandwiches to make (N_to_make)
        N_allergic_unserved = len(unserved_allergic)
        N_non_allergic_unserved = len(unserved_non_allergic)

        # GF sandwiches needed for allergic children
        gf_to_make = max(0, N_allergic_unserved - avail_gf_sandwiches)

        # Sandwiches needed for non-allergic children.
        # They can use any available sandwich.
        # Available for them are all regular sandwiches MINUS those GF sandwiches
        # that *must* be used for allergic children.
        available_any_for_non_allergic = avail_reg_sandwiches - min(avail_gf_sandwiches, N_allergic_unserved)

        non_gf_to_make = max(0, N_non_allergic_unserved - available_any_for_non_allergic)

        N_to_make = gf_to_make + non_gf_to_make

        # 10. Count sandwiches currently in the kitchen (for 'put_on_tray' action)
        N_kitchen_sandwiches = len(sandwiches_in_kitchen)

        # 11. Identify places with waiting unserved children
        places_with_waiting_unserved_children = {
            self.waiting_places.get(child) for child in unserved_children if child in self.waiting_places # Ensure child has a waiting place defined
        }
        places_with_waiting_unserved_children.discard(None) # Remove None if any child didn't have a waiting place

        # 12. Identify places with trays
        places_with_trays = {
            get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("tray")
        }

        # 13. Count places needing a tray move
        N_places_need_tray = len(places_with_waiting_unserved_children - places_with_trays)

        # 14. Calculate total heuristic
        # Cost = Serve + Move Tray + Put on Tray (existing) + Make + Put on Tray (newly made)
        # Cost = N_unserved + N_places_need_tray + N_kitchen_sandwiches + N_to_make + N_to_make
        total_heuristic = N_unserved + N_places_need_tray + N_kitchen_sandwiches + 2 * N_to_make

        return total_heuristic
