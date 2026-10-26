from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(waiting child1 table1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnack15Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches.
    It considers the number of children waiting, the number of sandwiches available, and whether gluten-free
    sandwiches are needed.

    # Assumptions
    - Each child needs one sandwich.
    - Sandwiches can be made at the kitchen and placed on a tray.
    - Trays can be moved to different places.
    - Gluten-free sandwiches are made only if there are allergic children waiting.

    # Heuristic Initialization
    - Extract information about allergic children, waiting children, available bread/content, and trays from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of waiting children.
    2. Count the number of sandwiches that still need to be created.
    3. Check if there are allergic children waiting.
    4. Estimate the number of actions required:
       - For each child, we need to make a sandwich (if it doesn't exist), put it on a tray, and serve it.
       - If there are allergic children, prioritize making gluten-free sandwiches for them.
       - Account for the actions of making sandwiches, putting them on trays, moving trays, and serving children.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting relevant information from the task."""
        self.goals = task.goals
        static_facts = task.static

        # Extract information about allergic children.
        self.allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")
        }

        # Extract information about not allergic children.
        self.not_allergic_children = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")
        }

        # Extract information about waiting children and their locations.
        self.waiting_children = {
            get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "waiting", "*", "*")
        }

        # Extract information about no_gluten bread and content
        self.no_gluten_bread = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")
        }

        self.no_gluten_content = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")
        }

        # Extract information about trays and their locations.
        # self.trays = {get_parts(fact)[1] for fact in static_facts if match(fact, "at", "*", "*")}

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Check if the goal is already reached.
        if task.goal_reached(None, state):
            return 0

        # Count the number of waiting children who have not been served.
        unserved_children = 0
        for child in self.waiting_children:
            if f"(served {child})" not in state:
                unserved_children += 1

        # If all children are served, the heuristic value is 0.
        if unserved_children == 0:
            return 0

        # Count the number of sandwiches that need to be made.
        needed_sandwiches = unserved_children

        # Check if there are allergic children waiting.
        allergic_children_waiting = False
        for child in self.allergic_children:
            if f"(served {child})" not in state and child in self.waiting_children:
                allergic_children_waiting = True
                break

        # Estimate the number of actions required.
        heuristic_value = 0

        # Actions per child: make sandwich (if needed), put on tray, move tray (if needed), serve.
        heuristic_value += needed_sandwiches * 3  # make, put on tray, serve

        # Account for tray movements.
        # This is a simplification; a more accurate heuristic would consider the tray locations.
        # heuristic_value += 1  # Assuming at least one tray movement is needed.

        # If there are allergic children, prioritize making gluten-free sandwiches.
        if allergic_children_waiting:
            heuristic_value += 1  # Making gluten-free sandwiches might require extra steps.

        return heuristic_value
