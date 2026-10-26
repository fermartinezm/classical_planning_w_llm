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


class childsnack10Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches,
    considering gluten allergies and the need to make sandwiches and move trays.

    # Assumptions
    - Each child needs one sandwich.
    - Sandwiches must be made before being put on a tray.
    - Trays must be at the location of the child to serve them.
    - The heuristic doesn't account for the limited number of bread/content portions.

    # Heuristic Initialization
    - Identify children who are allergic to gluten.
    - Identify children who are not allergic to gluten.
    - Store the locations of waiting children.
    - Identify trays and their initial locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children who are waiting and not yet served.
    2. For each waiting child:
       a. Determine if the child is allergic to gluten.
       b. Estimate the cost to serve the child:
          - If a gluten-free sandwich is needed, estimate the cost to make one (if one doesn't exist).
          - Estimate the cost to put the sandwich on a tray (if it's not already).
          - Estimate the cost to move the tray to the child's location (if it's not already there).
          - Add the cost to serve the sandwich.
    3. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting information about children,
        allergies, and tray locations from the static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")
        }
        self.waiting_children_locations = {
            get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "waiting", "*", "*")
        }
        self.tray_locations = {}
        self.trays = set()
        for fact in static_facts:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if "tray" in parts[1]:
                    self.tray_locations[parts[1]] = parts[2]
                    self.trays.add(parts[1])

    def __call__(self, node):
        """Estimate the minimum cost to serve all waiting children."""
        state = node.state

        # Check if the goal is already reached
        if all(goal in state for goal in self.goals):
            return 0

        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        waiting_children = set(self.waiting_children_locations.keys())
        unserved_children = waiting_children - served_children

        total_cost = 0
        for child in unserved_children:
            child_location = self.waiting_children_locations[child]
            is_allergic = child in self.allergic_children

            # Estimate cost to make a sandwich (if needed)
            sandwich_needed = True
            for fact in state:
                if match(fact, "ontray", "*", "*"):
                    sandwich = get_parts(fact)[1]
                    if is_allergic and match(fact, "no_gluten_sandwich", sandwich):
                        sandwich_needed = False
                        break
                    elif not is_allergic and not any(match(fact, "no_gluten_sandwich", s) for s in state if match(fact, "no_gluten_sandwich", "*")):
                        sandwich_needed = False
                        break

            if sandwich_needed:
                total_cost += 2  # Estimate cost to make a sandwich (make_sandwich or make_sandwich_no_gluten)

            # Estimate cost to put the sandwich on a tray (if needed)
            sandwich_on_tray = False
            for fact in state:
                if match(fact, "ontray", "*", "*"):
                    sandwich = get_parts(fact)[1]
                    if is_allergic and match(fact, "no_gluten_sandwich", sandwich):
                        sandwich_on_tray = True
                        break
                    elif not is_allergic and not any(match(fact, "no_gluten_sandwich", s) for s in state if match(fact, "no_gluten_sandwich", "*")):
                        sandwich_on_tray = True
                        break

            if not sandwich_on_tray:
                total_cost += 1  # Estimate cost to put sandwich on tray (put_on_tray)

            # Estimate cost to move the tray to the child's location (if needed)
            tray_at_location = False
            for tray in self.trays:
                for fact in state:
                    if match(fact, "at", tray, child_location):
                        tray_at_location = True
                        break
                if tray_at_location:
                    break

            if not tray_at_location:
                total_cost += 1  # Estimate cost to move tray (move_tray)

            total_cost += 1  # Cost to serve the sandwich (serve_sandwich or serve_sandwich_no_gluten)

        return total_cost
