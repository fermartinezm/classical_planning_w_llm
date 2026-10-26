# Helper functions to parse PDDL facts
def get_parts(fact):
    """Splits a PDDL fact string into its predicate and arguments."""
    # Example: '(at tray1 kitchen)' -> ['at', 'tray1', 'kitchen']
    # Example: '(served child1)' -> ['served', 'child1']
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a fact matches a pattern of predicate and arguments."""
    # Example: match('(at tray1 kitchen)', 'at', '*', 'kitchen') -> True
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

from fnmatch import fnmatch
# Assuming Heuristic base class is available at this path
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions required to reach a goal state
    by summing up the estimated costs for:
    1. Serving unserved children.
    2. Making necessary sandwiches.
    3. Putting necessary sandwiches on trays.
    4. Moving trays to locations needing deliveries.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static task information.

        Heuristic Initialization:
        - Stores the goal facts (served children).
        - Parses static facts to determine allergy status and waiting locations
          for all children defined in the problem.
        """
        self.goals = task.goals
        static_facts = task.static

        # Map child name to allergy status (True if allergic_gluten)
        self.child_allergy = {}
        # Map child name to waiting place
        self.child_location = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                child_name = parts[1]
                self.child_allergy[child_name] = True
            elif parts[0] == 'not_allergic_gluten':
                child_name = parts[1]
                self.child_allergy[child_name] = False
            elif parts[0] == 'waiting':
                child_name = parts[1]
                place_name = parts[2]
                self.child_location[child_name] = place_name

        # Identify all children that need to be served in the goal
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'served'}


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Summary:
        The heuristic estimates the remaining cost by summing four components:
        1. The number of unserved children (representing the final 'serve' actions).
        2. The number of sandwiches that still need to be made to satisfy the needs
           of unserved children.
        3. The number of sandwiches that need to be put on trays to be ready for delivery.
        4. The number of distinct locations needing sandwich deliveries that do not
           currently have a tray present.

        Assumptions:
        - Tray capacity is effectively unlimited for heuristic calculation purposes.
        - Sufficient bread, content, and 'notexist' sandwich objects are available
          to make any required sandwiches.
        - A sandwich is gluten-free if and only if the predicate
          '(no_gluten_sandwich s)' is true for that sandwich 's' in the state.
          Otherwise, it is considered a regular sandwich.
        - Allergy status and waiting location for children are available in static facts.

        Step-By-Step Thinking for Computing Heuristic:
        1. Identify the set of children who are in the goal state (need to be served)
           but are not yet marked as served in the current state. Let this count be H1.
        2. Determine the type of sandwich (regular or gluten-free) required for each
           unserved child based on their allergy status (from static facts).
        3. Count the total number of regular and gluten-free sandwiches needed for
           all unserved children.
        4. Count the total number of regular and gluten-free sandwiches currently
           available in the state (either 'at_kitchen_sandwich' or 'ontray').
           Determine sandwich type by checking for the '(no_gluten_sandwich s)' predicate.
        5. Calculate the number of sandwiches of each type that still need to be made:
           `max(0, needed_count - available_count)`. Sum these counts to get H2.
        6. Count the total number of sandwiches currently on trays ('ontray ?s ?t').
        7. Calculate the number of sandwiches that need to transition to being on a tray.
           This is the total number of sandwiches needed for unserved children minus
           the number of sandwiches already on trays, capped at zero. Let this be H3.
        8. Identify all unique locations where unserved children are waiting.
        9. For each such location 'p', calculate the number of *additional* suitable
           sandwiches needed on trays *at that specific location*. This is the count
           of unserved children at 'p' minus the count of suitable sandwiches already
           'ontray' at 'p', capped at zero, summed over both sandwich types.
        10. Count the number of locations 'p' where additional sandwiches are needed
            on trays ('Add_needed_at_p > 0') AND there is no tray currently located
            at 'p' ('at ?t p' is false for all trays 't'). Let this count be H4.
        11. The total heuristic value is the sum H1 + H2 + H3 + H4.
        """
        state = node.state

        # H1: Number of unserved children (cost of serve actions)
        unserved_children = {
            child for child in self.goal_children
            if f'(served {child})' not in state
        }
        h1 = len(unserved_children)

        if h1 == 0:
            return 0 # Goal state

        # Group unserved children by location and allergy status
        unserved_by_location_allergy = {} # {(location, is_allergic): [child1, ...]}
        total_unserved_regular = 0
        total_unserved_gluten_free = 0

        for child in unserved_children:
            location = self.child_location.get(child)
            is_allergic = self.child_allergy.get(child, False)
            if location: # Only consider children whose location is known
                key = (location, is_allergic)
                unserved_by_location_allergy.setdefault(key, []).append(child)
                if is_allergic:
                    total_unserved_gluten_free += 1
                else:
                    total_unserved_regular += 1

        # Count available sandwiches (at_kitchen or ontray) and their types
        available_sandwiches = set()
        available_regular_sandwiches = set()
        available_gluten_free_sandwiches = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at_kitchen_sandwich':
                s_name = parts[1]
                available_sandwiches.add(s_name)
            elif parts[0] == 'ontray':
                s_name = parts[1]
                available_sandwiches.add(s_name)

        # Determine allergy status of available sandwiches
        for s_name in available_sandwiches:
             if f'(no_gluten_sandwich {s_name})' in state:
                 available_gluten_free_sandwiches.add(s_name)
             else:
                 # Assume regular if not explicitly marked no_gluten
                 available_regular_sandwiches.add(s_name)

        # H2: Number of sandwiches that must be made (cost of make actions)
        num_available_regular = len(available_regular_sandwiches)
        num_available_gluten_free = len(available_gluten_free_sandwiches)

        to_make_regular = max(0, total_unserved_regular - num_available_regular)
        to_make_gluten_free = max(0, total_unserved_gluten_free - num_available_gluten_free)
        h2 = to_make_regular + to_make_gluten_free

        # H3: Number of sandwiches that need to become ontray (cost of put_on_tray actions)
        num_ontray = sum(1 for fact in state if match(fact, 'ontray', '*', '*'))
        total_needed_ontray = total_unserved_regular + total_unserved_gluten_free
        h3 = max(0, total_needed_ontray - num_ontray)

        # H4: Number of locations needing deliveries that don't have a tray (cost of initial move_tray to location)
        h4 = 0
        locations_needing_delivery = set()
        needed_at_location = {} # {location: {allergy_status: count}}
        ontray_at_location = {} # {location: {allergy_status: count}}

        # 1. Count needed sandwiches per location and allergy
        for (location, is_allergic), children_list in unserved_by_location_allergy.items():
            needed_at_location.setdefault(location, {})[is_allergic] = len(children_list)
            locations_needing_delivery.add(location)

        # 2. Count suitable sandwiches already ontray per location and allergy
        for fact in state:
            if match(fact, 'ontray', '*', '*'):
                s_name = get_parts(fact)[1]
                t_name = get_parts(fact)[2]
                s_is_gluten_free = f'(no_gluten_sandwich {s_name})' in state

                # Find the location of the tray
                tray_location = None
                for tray_fact in state:
                    if match(tray_fact, 'at', t_name, '*'):
                        tray_location = get_parts(tray_fact)[2]
                        break

                if tray_location:
                     ontray_at_location.setdefault(tray_location, {})[s_is_gluten_free] = \
                         ontray_at_location.setdefault(tray_location, {}).get(s_is_gluten_free, 0) + 1

        # 3. Calculate Add_needed_at_p and H4
        for location in locations_needing_delivery:
            needed_reg = needed_at_location.get(location, {}).get(False, 0)
            needed_gf = needed_at_location.get(location, {}).get(True, 0)

            ontray_reg = ontray_at_location.get(location, {}).get(False, 0)
            ontray_gf = ontray_at_location.get(location, {}).get(True, 0)

            add_needed_at_p_reg = max(0, needed_reg - ontray_reg)
            add_needed_at_p_gf = max(0, needed_gf - ontray_gf)

            add_needed_at_p = add_needed_at_p_reg + add_needed_at_p_gf

            if add_needed_at_p > 0:
                # Check if there is any tray at this location
                tray_at_p = any(match(fact, 'at', '*', location) for fact in state)
                if not tray_at_p:
                    h4 += 1 # Need to move a tray here

        return h1 + h2 + h3 + h4
