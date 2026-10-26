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


class childsnack12Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches.
    It considers the actions of making sandwiches, putting them on trays, moving trays to the children, and serving the sandwiches.

    # Assumptions
    - Each child needs one sandwich.
    - Making a sandwich requires bread and content.
    - Trays can hold multiple sandwiches.
    - Trays need to be at the same place as the child to serve the sandwich.

    # Heuristic Initialization
    - Count the number of children waiting to be served.
    - Identify children with gluten allergies.
    - Identify available bread and content portions, distinguishing between gluten-free and regular options.
    - Identify the locations of the trays.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children waiting to be served.
    2. Count the number of sandwiches that need to be made.
    3. Estimate the number of 'make_sandwich' actions required.
       - If there are gluten-allergic children, prioritize making gluten-free sandwiches.
       - Otherwise, make regular sandwiches.
    4. Estimate the number of 'put_on_tray' actions required.
    5. Estimate the number of 'move_tray' actions required.
    6. Estimate the number of 'serve_sandwich' actions required.
       - Distinguish between 'serve_sandwich_no_gluten' and 'serve_sandwich' actions based on allergies.
    7. Sum up the estimated number of actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting relevant information from the task.
        """
        self.goals = task.goals
        static_facts = task.static

        # Count children and identify those with gluten allergies.
        self.allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")
        }
        self.waiting_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "waiting", "*", "*")
        }

        # Identify available bread and content.
        self.gluten_free_bread = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")
        }
        self.gluten_free_content = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")
        }

    def __call__(self, node):
        """
        Estimate the number of actions required to reach the goal state from the given state.
        """
        state = node.state
        if self.goal_reached(state):
            return 0

        # Count the number of children waiting and served.
        waiting_children_in_state = sum(1 for child in self.waiting_children if any(match(fact, "waiting", child, "*") for fact in state))
        served_children_in_state = sum(1 for fact in state if match(fact, "served", "*"))
        unserved_children = waiting_children_in_state - served_children_in_state

        # Count available bread and content at the kitchen.
        available_bread = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*"))
        available_content = sum(1 for fact in state if match(fact, "at_kitchen_content", "*"))

        # Count available gluten-free bread and content at the kitchen.
        available_gluten_free_bread = sum(1 for bread in self.gluten_free_bread if any(match(fact, "at_kitchen_bread", bread) for fact in state))
        available_gluten_free_content = sum(1 for content in self.gluten_free_content if any(match(fact, "at_kitchen_content", content) for fact in state))

        # Count available trays at the kitchen
        available_trays_kitchen = sum(1 for fact in state if match(fact, "at", "*", "kitchen"))

        # Count available sandwiches on trays
        sandwiches_on_trays = sum(1 for fact in state if match(fact, "ontray", "*", "*"))

        # Estimate the number of make_sandwich actions needed.
        make_sandwich_actions = 0
        gluten_free_sandwiches_needed = 0
        regular_sandwiches_needed = 0

        # Calculate how many gluten-free sandwiches are needed
        for child in self.allergic_children:
            if any(match(fact, "waiting", child, "*") for fact in state) and not any(match(fact, "served", child) for fact in state):
                gluten_free_sandwiches_needed += 1

        # Calculate how many regular sandwiches are needed
        for child in self.not_allergic_children:
            if any(match(fact, "waiting", child, "*") for fact in state) and not any(match(fact, "served", child) for fact in state):
                regular_sandwiches_needed += 1

        # Prioritize making gluten-free sandwiches if needed
        if gluten_free_sandwiches_needed > 0:
            make_sandwich_actions += gluten_free_sandwiches_needed
        else:
            make_sandwich_actions += regular_sandwiches_needed

        # Estimate the number of put_on_tray actions needed.
        put_on_tray_actions = max(0, make_sandwich_actions - sandwiches_on_trays)

        # Estimate the number of move_tray actions needed.
        move_tray_actions = 0
        waiting_children_locations = {}
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                child = parts[1]
                location = parts[2]
                waiting_children_locations[child] = location

        tray_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                tray = parts[1]
                location = parts[2]
                tray_locations[tray] = location

        # Estimate the number of serve_sandwich actions needed.
        serve_sandwich_actions = unserved_children

        # Total estimated actions.
        total_actions = make_sandwich_actions + put_on_tray_actions + move_tray_actions + serve_sandwich_actions

        return total_actions
