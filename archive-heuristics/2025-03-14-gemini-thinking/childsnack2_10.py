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
    This heuristic estimates the minimum number of actions required to serve all waiting children.
    It considers the necessary steps: making sandwiches, putting them on trays, moving trays to the children's locations, and serving the sandwiches, while also accounting for gluten allergies.

    # Assumptions:
    - We assume that for each waiting child, a sandwich needs to be prepared and served.
    - We prioritize using existing sandwiches on trays if available.
    - We assume that bread and content are always available in the kitchen if needed to make a sandwich.
    - Moving trays is necessary if a suitable sandwich is on a tray but not at the child's location.

    # Heuristic Initialization
    - Identify children who are allergic and not allergic to gluten from the static facts.
    - Identify children who are waiting and their waiting places from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For each child that is currently waiting and not yet served:
    1. Determine if the child is allergic to gluten.
    2. Check if there is a suitable sandwich already on a tray at the child's waiting place.
       - If yes, the estimated cost is 1 (serve_sandwich or serve_sandwich_no_gluten).
    3. If not, check if there is a suitable sandwich on any tray in the kitchen.
       - If yes, the estimated cost is 2 (move_tray to child's place + serve_sandwich or serve_sandwich_no_gluten).
    4. If no suitable sandwich is on any tray, estimate the cost of making a sandwich, putting it on a tray, moving the tray, and serving.
       - The estimated cost is 4 (make_sandwich[_no_gluten] + put_on_tray + move_tray + serve_sandwich[_no_gluten]).
    5. Sum up the estimated costs for all unserved children to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts about children's allergies and waiting status.
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
        Estimate the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0

        served_children = set()
        sandwiches_on_tray = {}  # {tray: [sandwiches]}
        tray_locations = {} # {tray: location}
        kitchen_sandwiches = set()

        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])
            elif match(fact, "ontray", "*", "*"):
                sandwich = get_parts(fact)[1]
                tray = get_parts(fact)[2]
                if tray not in sandwiches_on_tray:
                    sandwiches_on_tray[tray] = []
                sandwiches_on_tray[tray].append(sandwich)
            elif match(fact, "at", "*", "*"):
                tray = get_parts(fact)[1]
                location = get_parts(fact)[2]
                tray_locations[tray] = location
            elif match(fact, "at_kitchen_sandwich", "*"):
                kitchen_sandwiches.add(get_parts(fact)[1])


        for child in self.waiting_children_places:
            if child not in served_children:
                waiting_place = self.waiting_children_places[child]
                is_allergic = child in self.allergic_children

                served = False
                # Check for suitable sandwich on tray at the waiting place
                for tray, location in tray_locations.items():
                    if location == waiting_place and tray in sandwiches_on_tray:
                        for sandwich in sandwiches_on_tray[tray]:
                            sandwich_no_gluten = "(no_gluten_sandwich {})".format(sandwich) in state
                            if is_allergic and sandwich_no_gluten:
                                heuristic_value += 1 # Serve action
                                served = True
                                break
                            elif not is_allergic and not sandwich_no_gluten:
                                heuristic_value += 1 # Serve action
                                served = True
                                break
                    if served:
                        break

                if not served:
                    # Check for suitable sandwich on tray in the kitchen
                    for tray, location in tray_locations.items():
                        if location == 'kitchen' and tray in sandwiches_on_tray:
                            for sandwich in sandwiches_on_tray[tray]:
                                sandwich_no_gluten = "(no_gluten_sandwich {})".format(sandwich) in state
                                if is_allergic and sandwich_no_gluten:
                                    heuristic_value += 2 # move_tray + serve
                                    served = True
                                    break
                                elif not is_allergic and not sandwich_no_gluten:
                                    heuristic_value += 2 # move_tray + serve
                                    served = True
                                    break
                        if served:
                            break

                if not served:
                    heuristic_value += 4 # make_sandwich + put_on_tray + move_tray + serve

        return heuristic_value
