from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments, unless args contains wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting
    children. It counts the necessary actions in the sandwich delivery pipeline:
    making sandwiches, putting them on trays, moving trays to children's
    locations, and finally serving the sandwiches.

    # Heuristic Components
    The heuristic value is the sum of the following estimated costs:
    1.  Serving Cost: One action per unserved child (the 'serve' action).
    2.  Tray Placement Cost: One action for each location (other than the kitchen)
        where unserved children are waiting but no tray is currently present
         (a 'move_tray' action is needed).
    3.  Put-on-Tray Cost: One action for each sandwich that needs to be put on
        a tray to serve an unserved child, but is not yet on a tray. This is
        estimated as the number of unserved children minus the number of
        sandwiches already on trays.
    4.  Make Sandwich Cost: One action for each sandwich that needs to be made
        to serve an unserved child, but does not yet exist (is 'notexist').
        This is estimated as the number of unserved children minus the number
        of sandwiches already made (in kitchen or on trays), capped by the
        number of available 'notexist' sandwich objects.

    # Assumptions and Simplifications
    - Assumes sufficient ingredients (bread, content) are available in the
      kitchen to make needed sandwiches.
    - Assumes trays are available somewhere to be moved or used for 'put_on_tray'.
    - Does not distinguish between gluten-free and regular sandwiches/needs
      when counting 'put_on_tray' and 'make_sandwich' actions, simplifying
      the pipeline count. It assumes any available sandwich can eventually
      satisfy a need up to the total counts.
    - Ignores potential conflicts or resource limitations beyond the count
      of 'notexist' sandwich objects.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting initial waiting children and
        their locations/allergies from static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Pre-process static facts to find initial waiting children and their locations/allergies
        # We only care about children who are initially waiting, as they are the ones
        # who will appear in the goal state as 'served'.
        self.initial_waiting_children = {} # {child: place}
        self.allergic_children_static = set() # {child} - Store from static

        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                _, child, place = get_parts(fact)
                self.initial_waiting_children[child] = place
            elif match(fact, "allergic_gluten", "*"):
                _, child = get_parts(fact)
                self.allergic_children_static.add(child)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to serve
        all initially waiting children who are not yet served.
        """
        state = node.state
        h = 0

        # 1. Count unserved children and identify places needing service
        unserved_children_info = {} # {child: place}
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Iterate through children who were initially waiting
        for child, place in self.initial_waiting_children.items():
            # If a child was initially waiting but is not yet served in the current state
            if child not in served_children:
                unserved_children_info[child] = place

        n_unserved = len(unserved_children_info)

        # If all children are served, the heuristic is 0
        if n_unserved == 0:
            return 0

        # Component 1: Serving Cost
        # Each unserved child needs one 'serve' action.
        h += n_unserved

        # 2. Tray Placement Cost
        # Identify all distinct places where unserved children are waiting
        places_with_unserved = set(unserved_children_info.values())

        # Identify all places where trays are currently located
        trays_at_place = {get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")}

        # Component 2: Tray Movement Cost
        # For each place (other than the kitchen) that has unserved children
        # but no tray, we estimate one 'move_tray' action is needed.
        for place in places_with_unserved:
            if place != 'kitchen' and place not in trays_at_place:
                 h += 1

        # 3. Put-on-Tray Cost
        # Count sandwiches already on trays in the current state
        sandwiches_on_trays = {get_parts(fact)[1] for fact in state if match(fact, "ontray", "*", "*")}
        n_sandwiches_on_trays = len(sandwiches_on_trays)

        # Component 3: Put-on-Tray Cost
        # Estimate the number of sandwiches that need to be put on a tray.
        # This is the total number of sandwiches needed (equal to unserved children)
        # minus those already on trays.
        needed_put_on_tray = max(0, n_unserved - n_sandwiches_on_trays)
        h += needed_put_on_tray

        # 4. Make Sandwich Cost
        # Count sandwiches that have been made (are either in the kitchen or on a tray)
        sandwiches_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        sandwiches_made = sandwiches_kitchen | sandwiches_on_trays # Union of sets
        n_sandwiches_made = len(sandwiches_made)

        # Count sandwich objects that have not yet been made ('notexist')
        n_notexist = sum(1 for fact in state if match(fact, "notexist", "*"))

        # Component 4: Make Sandwich Cost
        # Estimate the number of sandwiches that need to be made.
        # This is the total number of sandwiches needed (equal to unserved children)
        # minus those already made. This count is capped by the number of
        # available 'notexist' sandwich objects.
        needed_make_sandwich = max(0, n_unserved - n_sandwiches_made)
        h += min(n_notexist, needed_make_sandwich)

        return h

