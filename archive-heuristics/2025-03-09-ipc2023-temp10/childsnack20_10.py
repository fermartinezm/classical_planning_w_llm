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


class childsnack20Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches,
    considering gluten allergies and the need to make sandwiches and move trays.

    # Assumptions
    - Each child needs one sandwich.
    - Sandwiches can be made with or without gluten.
    - Allergic children must receive gluten-free sandwiches.
    - Trays can be moved between places.
    - Bread and content are available at the kitchen.

    # Heuristic Initialization
    - Extract information about children's allergies, waiting children, available bread/content, and the initial tray locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of unserved children.
    2. For each unserved child:
       - Determine if the child is allergic to gluten.
       - If allergic, check if a gluten-free sandwich is available on a tray at the child's location. If not, estimate the cost of making one and putting it on a tray.
       - If not allergic, check if any sandwich is available on a tray at the child's location. If not, estimate the cost of making one and putting it on a tray.
    3. Estimate the cost of moving trays to the locations where children are waiting.
    4. Sum up the estimated costs for all children to get the overall heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting relevant information from the task."""
        self.goals = task.goals
        self.static = task.static

        # Extract information about children's allergies.
        self.allergic_children = {
            get_parts(fact)[1] for fact in self.static if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children = {
            get_parts(fact)[1] for fact in self.static if match(fact, "not_allergic_gluten", "*")
        }

        # Extract information about waiting children and their locations.
        self.waiting_children = {
            get_parts(fact)[1]: get_parts(fact)[2] for fact in self.static if match(fact, "waiting", "*", "*")
        }

        # Extract information about available gluten-free bread and content.
        self.no_gluten_bread = {
            get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_bread", "*")
        }
        self.no_gluten_content = {
            get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_content", "*")
        }

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Check if the state is a goal state.
        if self.goal_reached(state):
            return 0

        # Count the number of served children.
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Count the number of unserved children.
        unserved_children = set(self.waiting_children.keys()) - served_children

        # Estimate the cost for each unserved child.
        total_cost = 0
        for child in unserved_children:
            child_location = self.waiting_children[child]
            is_allergic = child in self.allergic_children

            # Check if a suitable sandwich is available on a tray at the child's location.
            sandwich_available = False
            for fact in state:
                if match(fact, "ontray", "*", "*"):
                    sandwich = get_parts(fact)[1]
                    tray = get_parts(fact)[2]
                    for fact2 in state:
                        if match(fact2, "at", tray, child_location):
                            if is_allergic:
                                if any(match(fact3, "no_gluten_sandwich", sandwich) for fact3 in state):
                                    sandwich_available = True
                                    break
                            else:
                                sandwich_available = True
                                break
                    if sandwich_available:
                        break

            # If no suitable sandwich is available, estimate the cost of making one and putting it on a tray.
            if not sandwich_available:
                # Estimate the cost of making a sandwich (make_sandwich or make_sandwich_no_gluten).
                sandwich_cost = 3  # Making sandwich, putting on tray
                total_cost += sandwich_cost

            # Estimate the cost of moving the tray to the child's location (move_tray).
            tray_at_location = False
            for fact in state:
                if match(fact, "at", "*", child_location):
                    tray_at_location = True
                    break
            if not tray_at_location:
                total_cost += 1  # Moving tray

        return total_cost

    def goal_reached(self, state):
        """
        Check if the goal has been reached.

        @return True if all the goals are reached, False otherwise
        """
        return self.goals <= state
