from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnack4Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches,
    considering gluten allergies and the need to make and place sandwiches on trays.

    # Assumptions
    - Each child needs one sandwich.
    - Sandwiches must be made before being put on a tray.
    - Trays must be at the same place as the child to serve the sandwich.
    - The number of actions is estimated based on the number of children that still need to be served,
      the number of sandwiches that need to be made, and the number of trays that need to be moved.

    # Heuristic Initialization
    - Identify children with gluten allergies.
    - Store the locations of waiting children.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children who are waiting and not yet served.
    2. Count the number of sandwiches on trays.
    3. Count the number of sandwiches that still need to be made, considering gluten allergies.
    4. Estimate the number of trays that need to be moved to serve the children.
    5. Sum the estimated costs for making sandwiches, putting them on trays, moving trays, and serving children.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Children with gluten allergies.
        - Locations of waiting children.
        """
        self.goals = task.goals
        static_facts = task.static

        # Identify children with gluten allergies.
        self.allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")
        }

        # Identify locations of waiting children.
        self.waiting_children_locations = {
            get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "waiting", "*", "*")
        }

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Count the number of served children.
        served_children_count = sum(1 for fact in state if match(fact, "served", "*"))

        # Count the total number of children.
        total_children_count = len(self.waiting_children_locations)

        # Calculate the number of children who still need to be served.
        unserved_children_count = total_children_count - served_children_count

        if unserved_children_count == 0:
            return 0  # Goal state reached

        # Count the number of sandwiches on trays.
        sandwiches_on_trays_count = sum(1 for fact in state if match(fact, "ontray", "*", "*"))

        # Count the number of sandwiches that still need to be made.
        sandwiches_to_make_count = max(0, unserved_children_count - sandwiches_on_trays_count)

        # Estimate the number of trays that need to be moved.
        trays_to_move_count = 0
        trays = set()
        children_served = set()
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                sandwich, tray = get_parts(fact)[1], get_parts(fact)[2]
                trays.add(tray)
            if match(fact, "served", "*"):
                child = get_parts(fact)[1]
                children_served.add(child)

        children_to_serve = set(self.waiting_children_locations.keys()) - children_served
        tray_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                tray, location = get_parts(fact)[1], get_parts(fact)[2]
                if tray in trays:
                    tray_locations[tray] = location

        for child in children_to_serve:
            child_location = self.waiting_children_locations[child]
            tray_at_child_location = False
            for tray in trays:
                if tray in tray_locations and tray_locations[tray] == child_location:
                    tray_at_child_location = True
                    break
            if not tray_at_child_location:
                trays_to_move_count += 1

        # Estimate the total cost.
        total_cost = (
            sandwiches_to_make_count  # Cost for making sandwiches
            + sandwiches_on_trays_count # Cost for putting sandwiches on trays
            + trays_to_move_count  # Cost for moving trays
            + unserved_children_count  # Cost for serving children
        )

        return total_cost
