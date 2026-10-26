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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children with appropriate snacks.
    It considers the steps needed to make a sandwich, put it on a tray, move the tray to the child's location, and serve the sandwich,
    taking into account gluten allergies.

    # Assumptions:
    - There are always enough bread and content portions at the kitchen to make sandwiches.
    - Trays are available at the kitchen initially.
    - The goal is to serve all children who are initially waiting.

    # Heuristic Initialization
    - Identify children who are allergic and not allergic to gluten from the static facts.
    - Identify children who are waiting and their locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For each child who is currently waiting and not yet served:
    1. Determine if the child is allergic to gluten.
    2. Check if a suitable sandwich is already prepared and available at the correct location:
       - Look for a sandwich (gluten-free if allergic, regular if not) that is on a tray and the tray is at the child's waiting place.
       - If found, the estimated cost for this child is 1 (serve_sandwich action).
    3. If no suitable sandwich is at the correct location, check if a suitable sandwich is on a tray at the kitchen:
       - Look for a sandwich (gluten-free if allergic, regular if not) that is on a tray and the tray is at the kitchen.
       - If found, the estimated cost for this child is 2 (move_tray, serve_sandwich actions).
    4. If no suitable sandwich is on a tray, check if a suitable sandwich is already made at the kitchen:
       - Look for a sandwich (gluten-free if allergic, regular if not) that is at the kitchen.
       - If found, the estimated cost for this child is 3 (put_on_tray, move_tray, serve_sandwich actions).
    5. If no suitable sandwich is made, then we need to make a sandwich first:
       - The estimated cost for this child is 4 (make_sandwich, put_on_tray, move_tray, serve_sandwich actions).

    The heuristic value for a given state is the sum of the minimum estimated costs for serving each unserved waiting child.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting information about children's allergies and waiting status."""
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_children_places = {}

        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                self.allergic_children.add(get_parts(fact)[1])
            elif match(fact, "not_allergic_gluten", "*"):
                self.not_allergic_children.add(get_parts(fact)[1])
            elif match(fact, "waiting", "*", "*"):
                child = get_parts(fact)[1]
                place = get_parts(fact)[2]
                self.waiting_children_places[child] = place

    def __call__(self, node):
        """Estimate the number of actions needed to serve all waiting children."""
        state = node.state
        served_children = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])

        unserved_waiting_children = []
        for child in self.waiting_children_places:
            if child not in served_children:
                unserved_waiting_children.append(child)

        heuristic_value = 0
        for child in unserved_waiting_children:
            child_place = self.waiting_children_places[child]
            is_allergic = child in self.allergic_children

            sandwich_type_predicate = "no_gluten_sandwich" if is_allergic else "no_gluten_sandwich" # dummy, will be overwritten
            sandwich_type_check = "no_gluten_sandwich" if is_allergic else "not_no_gluten_sandwich"

            min_cost = 4  # Default cost: make, put on tray, move tray, serve

            # Check if suitable sandwich is already served (goal state) - should be 0 if goal already reached, but handled by search algorithm
            # Check if suitable sandwich is on tray at correct location
            for fact in state:
                if is_allergic:
                    if match(fact, "serve_sandwich_no_gluten", "*", child, "*", child_place): # not a state fact, but action name
                        min_cost = min(min_cost, 1) # Serve action
                    if match(fact, "ontray", "*", "*"):
                        sandwich = get_parts(fact)[1]
                        tray = get_parts(fact)[2]
                        for fact2 in state:
                            if match(fact2, "at", tray, child_place) and "(no_gluten_sandwich " + sandwich + ")" in state:
                                min_cost = min(min_cost, 1) # Serve action if sandwich is on tray at correct place
                else:
                    if match(fact, "serve_sandwich", "*", child, "*", child_place): # not a state fact, but action name
                        min_cost = min(min_cost, 1) # Serve action
                    if match(fact, "ontray", "*", "*"):
                        sandwich = get_parts(fact)[1]
                        tray = get_parts(fact)[2]
                        for fact2 in state:
                            if match(fact2, "at", tray, child_place) and "(no_gluten_sandwich " + sandwich + ")" not in state: # regular sandwich
                                min_cost = min(min_cost, 1) # Serve action if sandwich is on tray at correct place


            # Check if suitable sandwich is on tray at kitchen
            for fact in state:
                if is_allergic:
                    if match(fact, "ontray", "*", "*"):
                        sandwich = get_parts(fact)[1]
                        tray = get_parts(fact)[2]
                        if tray == "kitchen" and "(no_gluten_sandwich " + sandwich + ")" in state:
                            min_cost = min(min_cost, 2) # Move tray, serve
                else:
                    if match(fact, "ontray", "*", "*"):
                        sandwich = get_parts(fact)[1]
                        tray = get_parts(fact)[2]
                        if tray == "kitchen" and "(no_gluten_sandwich " + sandwich + ")" not in state: # regular sandwich
                            min_cost = min(min_cost, 2) # Move tray, serve

            # Check if suitable sandwich is at kitchen
            for fact in state:
                if is_allergic:
                    if match(fact, "at_kitchen_sandwich", "*"):
                        sandwich = get_parts(fact)[1]
                        if "(no_gluten_sandwich " + sandwich + ")" in state:
                            min_cost = min(min_cost, 3) # Put on tray, move tray, serve
                else:
                    if match(fact, "at_kitchen_sandwich", "*"):
                        sandwich = get_parts(fact)[1]
                        if "(no_gluten_sandwich " + sandwich + ")" not in state: # regular sandwich
                            min_cost = min(min_cost, 3) # Put on tray, move tray, serve


            heuristic_value += min_cost

        return heuristic_value
