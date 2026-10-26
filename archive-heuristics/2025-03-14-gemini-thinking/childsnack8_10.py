from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve all waiting children with snacks, considering their gluten allergies. It prioritizes serving children with existing sandwiches and then estimates the cost of making, traying, moving, and serving sandwiches.

    # Assumptions
    - There are always enough bread and content portions at the kitchen to make sandwiches.
    - Trays are always available at the kitchen initially.
    - The heuristic focuses on serving each child individually and sums up the estimated costs.

    # Heuristic Initialization
    - Extracts static information about children's gluten allergies and their waiting places from the task's static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For each child that is not yet served:
    1. Determine if the child is allergic to gluten and their waiting place.
    2. Check if a suitable sandwich (no-gluten if allergic) is already on a tray at the child's waiting place. If yes, the cost to serve this child is 1 (serve_sandwich or serve_sandwich_no_gluten).
    3. If not, check if a suitable sandwich is on a tray at the kitchen. If yes, the cost is 2 (move_tray + serve_sandwich/serve_sandwich_no_gluten).
    4. If not, check if a suitable sandwich is at the kitchen (not on a tray). If yes, the cost is 3 (put_on_tray + move_tray + serve_sandwich/serve_sandwich_no_gluten).
    5. If no suitable sandwich exists, estimate the cost to make a sandwich, put it on a tray, move the tray, and serve it. The cost is 4 (make_sandwich[_no_gluten] + put_on_tray + move_tray + serve_sandwich/serve_sandwich_no_gluten).
    The heuristic value for a state is the sum of the minimum costs calculated for each unserved child.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts about children's allergies and waiting places.
        """
        self.goals = task.goals
        static_facts = task.static

        self.child_allergies = {}
        self.child_waiting_places = {}

        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.child_allergies[child] = True
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.child_allergies[child] = False
            elif match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:3]
                self.child_waiting_places[child] = place

    def __call__(self, node):
        """
        Calculate the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0

        served_children = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])

        goal_children = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                goal_children.add(get_parts(goal)[1])

        unserved_children = goal_children - served_children

        for child in unserved_children:
            if child not in self.child_allergies: # In case child is not in static facts, assume not allergic
                is_allergic = False
            else:
                is_allergic = self.child_allergies[child]
            waiting_place = self.child_waiting_places[child]

            min_cost_for_child = 4  # Initialize with max possible cost (make, put, move, serve)

            # Check for suitable sandwich on tray at child's place
            for fact in state:
                if match(fact, "ontray", "*", "*"):
                    sandwich = get_parts(fact)[1]
                    tray = get_parts(fact)[2]
                    tray_at_place = False
                    sandwich_suitable = False

                    for tray_location_fact in state:
                        if match(tray_location_fact, "at", tray, waiting_place):
                            tray_at_place = True
                            break
                    if not tray_at_place:
                        continue

                    if is_allergic:
                        for sandwich_type_fact in state:
                            if match(sandwich_type_fact, "no_gluten_sandwich", sandwich):
                                sandwich_suitable = True
                                break
                    else:
                        sandwich_suitable = True # any sandwich is fine if not allergic

                    if sandwich_suitable and tray_at_place:
                        min_cost_for_child = min(min_cost_for_child, 1) # serve
                        break # Found a sandwich, no need to check further for this child
                if min_cost_for_child == 1:
                    break
            if min_cost_for_child == 1:
                heuristic_value += min_cost_for_child
                continue


            # Check for suitable sandwich on tray at kitchen
            for fact in state:
                if match(fact, "ontray", "*", "*"):
                    sandwich = get_parts(fact)[1]
                    tray = get_parts(fact)[2]
                    tray_at_kitchen = False
                    sandwich_suitable = False

                    for tray_location_fact in state:
                        if match(tray_location_fact, "at", tray, "kitchen"):
                            tray_at_kitchen = True
                            break
                    if not tray_at_kitchen:
                        continue

                    if is_allergic:
                        for sandwich_type_fact in state:
                            if match(sandwich_type_fact, "no_gluten_sandwich", sandwich):
                                sandwich_suitable = True
                                break
                    else:
                        sandwich_suitable = True

                    if sandwich_suitable and tray_at_kitchen:
                        min_cost_for_child = min(min_cost_for_child, 2) # move_tray + serve
                        break
                if min_cost_for_child <= 2:
                    break
            if min_cost_for_child <= 2:
                heuristic_value += min_cost_for_child
                continue


            # Check for suitable sandwich at kitchen (not on tray)
            for fact in state:
                if match(fact, "at_kitchen_sandwich", "*"):
                    sandwich = get_parts(fact)[1]
                    sandwich_suitable = False
                    if is_allergic:
                        for sandwich_type_fact in state:
                            if match(sandwich_type_fact, "no_gluten_sandwich", sandwich):
                                sandwich_suitable = True
                                break
                    else:
                        sandwich_suitable = True

                    if sandwich_suitable:
                        min_cost_for_child = min(min_cost_for_child, 3) # put_on_tray + move_tray + serve
                        break
                if min_cost_for_child <= 3:
                    break
            if min_cost_for_child <= 3:
                heuristic_value += min_cost_for_child
                continue


            # If no suitable sandwich found, need to make one
            heuristic_value += 4 # make_sandwich[_no_gluten] + put_on_tray + move_tray + serve

        return heuristic_value
