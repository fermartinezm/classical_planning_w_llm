from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robby rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve all waiting children with sandwiches,
    considering their gluten allergies and the need to make, tray, move, and serve sandwiches.

    # Assumptions:
    - Each waiting child needs to be served exactly one sandwich.
    - We prioritize serving children as quickly as possible.
    - Making a sandwich, putting it on a tray, moving the tray, and serving are distinct action steps.

    # Heuristic Initialization
    - Identify children who are allergic to gluten and those who are not from the static facts.
    - Identify children who are waiting and their waiting places from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For each child that is currently not served:
    1. Check if there is a suitable sandwich already on a tray at the child's waiting place.
       - If yes, the estimated cost is 1 (serve action).
    2. If not, check if there is a suitable sandwich already on a tray at the kitchen.
       - If yes, the estimated cost is 2 (move tray to child's place, serve action).
    3. If not, check if a suitable sandwich already exists at the kitchen (not on a tray).
       - If yes, the estimated cost is 3 (put sandwich on tray, move tray to child's place, serve action).
    4. If no suitable sandwich exists, we need to make one.
       - The estimated cost is 4 (make sandwich, put sandwich on tray, move tray to child's place, serve action).

    The heuristic value is the sum of the minimum estimated costs for each unserved child.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by pre-processing static facts to quickly access:
        - Allergy information for each child.
        - Waiting children and their locations.
        """
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
        """
        Calculate the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0

        served_children = set()
        sandwiches_on_tray_kitchen = []
        sandwiches_on_tray_place = {} # {place: [(sandwich, tray)]}
        sandwiches_at_kitchen = []

        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])
            elif match(fact, "ontray", "*", "*"):
                sandwich = get_parts(fact)[1]
                tray = get_parts(fact)[2]
                for fact2 in state:
                    if match(fact2, "at", "*", "*"):
                        tray_obj = get_parts(fact2)[1]
                        place = get_parts(fact2)[2]
                        if tray_obj == tray:
                            if place == 'kitchen':
                                sandwiches_on_tray_kitchen.append((sandwich, tray))
                            else:
                                if place not in sandwiches_on_tray_place:
                                    sandwiches_on_tray_place[place] = []
                                sandwiches_on_tray_place[place].append((sandwich, tray))
                            break # Found the location of the tray
                else: # Should not happen, tray location should always be defined
                    pass # Handle error if needed
            elif match(fact, "at_kitchen_sandwich", "*"):
                sandwiches_at_kitchen.append(get_parts(fact)[1])


        for child in self.waiting_children_places:
            if child not in served_children:
                child_place = self.waiting_children_places[child]
                is_allergic = child in self.allergic_children

                sandwich_served = False
                # 1. Check if there is a suitable sandwich already on a tray at the child's waiting place.
                if child_place in sandwiches_on_tray_place:
                    for sandwich, tray in sandwiches_on_tray_place[child_place]:
                        is_gluten_free_sandwich = "(no_gluten_sandwich {})".format(sandwich) in state
                        if (is_allergic and is_gluten_free_sandwich) or (not is_allergic and not is_gluten_free_sandwich):
                            heuristic_value += 1
                            sandwich_served = True
                            break # Serve with this sandwich
                    if sandwich_served:
                        continue # Move to next child

                # 2. Check if there is a suitable sandwich already on a tray at the kitchen.
                if not sandwich_served:
                    for sandwich, tray in sandwiches_on_tray_kitchen:
                        is_gluten_free_sandwich = "(no_gluten_sandwich {})".format(sandwich) in state
                        if (is_allergic and is_gluten_free_sandwich) or (not is_allergic and not is_gluten_free_sandwich):
                            heuristic_value += 2
                            sandwich_served = True
                            break
                    if sandwich_served:
                        continue

                # 3. Check if a suitable sandwich already exists at the kitchen (not on a tray).
                if not sandwich_served:
                    for sandwich in sandwiches_at_kitchen:
                        is_gluten_free_sandwich = "(no_gluten_sandwich {})".format(sandwich) in state
                        if (is_allergic and is_gluten_free_sandwich) or (not is_allergic and not is_gluten_free_sandwich):
                            heuristic_value += 3
                            sandwich_served = True
                            break
                    if sandwich_served:
                        continue

                # 4. If no suitable sandwich exists, we need to make one.
                if not sandwich_served:
                    heuristic_value += 4

        return heuristic_value
