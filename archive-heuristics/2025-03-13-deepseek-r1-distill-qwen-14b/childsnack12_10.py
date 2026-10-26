from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their sandwiches.

    # Assumptions:
    - Each child must be served exactly one sandwich.
    - A sandwich can be made if the required bread and content portions are available in the kitchen.
    - A sandwich must be placed on a tray before it can be served.
    - If a tray is not in the kitchen, it must be moved back before making more sandwiches.

    # Heuristic Initialization
    - Extracts static facts about which children are allergic to gluten and which are not.
    - Maps each child to their waiting place.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children who are waiting to be served.
    2. For each unserved child:
       a. Check if their sandwich is already made and on a tray in the correct place.
       b. If not, determine if the necessary ingredients are available.
       c. If the tray is not in the kitchen, add the cost to move it back.
       d. Estimate the steps needed to make the sandwich and serve it.
    3. Sum the estimated steps for all unserved children to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal conditions."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_place = {}
        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                self.allergic_children.add(get_parts(fact)[1])
            elif match(fact, "not_allergic_gluten", "*"):
                self.not_allergic_children.add(get_parts(fact)[1])
            elif match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1], get_parts(fact)[2]
                self.waiting_place[child] = place

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        def get_parts(fact):
            """Extract the components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, *args):
            """Check if a PDDL fact matches a given pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        def is_served(child):
            """Check if the child is served."""
            return f"(served {child})" in state

        def get_sandwich(child):
            """Find the sandwich associated with the child."""
            for fact in state:
                if match(fact, "ontray", "*", "*") and get_parts(fact)[1] == child:
                    return get_parts(fact)[0]
            return None

        def is_sandwich_ready(sandwich):
            """Check if the sandwich is ready and on a tray."""
            return f"(ontray {sandwich} * )" in state

        def tray_in_kitchen(tray):
            """Check if the tray is in the kitchen."""
            return f"(at {tray} kitchen)" in state

        total_cost = 0
        unserved_children = [child for child in self.waiting_place.keys() if not is_served(child)]

        for child in unserved_children:
            sandwich = get_sandwich(child)
            if sandwich is None:
                total_cost += 2  # Make sandwich and put on tray
            else:
                if not is_sandwich_ready(sandwich):
                    total_cost += 2  # Make sandwich and put on tray
                else:
                    if not tray_in_kitchen(get_parts(f"(ontray {sandwich} * )")[1]):
                        total_cost += 2  # Move tray back and serve
                    else:
                        total_cost += 1  # Just serve

        return total_cost
