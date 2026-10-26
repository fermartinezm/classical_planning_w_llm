from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions from example heuristics
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

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of unserved children and estimates the steps needed
    to make and deliver sandwiches to them.

    # Heuristic Components and Step-by-Step Thinking:
    The goal is to have `(served ?c)` for all children `?c`.
    To serve a child, a suitable sandwich must be on a tray at the child's location.
    The process for serving a child involves:
    1. Making a sandwich (if a suitable one isn't already made).
    2. Putting the sandwich on a tray (if it's not already on one).
    3. Moving the tray to the child's location (if it's not already there).
    4. Serving the sandwich to the child.

    The heuristic sums up the estimated minimum actions for these stages across all unserved children.

    1.  **Identify Unserved Children:** Count the number of children `?c` for whom `(served ?c)` is a goal but is not true in the current state. Each unserved child requires a final `serve` action.
    2.  **Estimate Sandwiches to Make:** Each unserved child needs a sandwich. Count the number of sandwiches that have already been made (either `at_kitchen_sandwich` or `ontray`). The number of *new* sandwiches that need to be made is the number of unserved children minus the number of sandwiches already made (clamped at zero). Each new sandwich requires a `make_sandwich` or `make_sandwich_no_gluten` action.
    3.  **Estimate Sandwiches to Put on Tray:** Each newly made sandwich needs to be put on a tray. This is roughly equal to the number of new sandwiches needed. Each requires a `put_on_tray` action.
    4.  **Estimate Tray Moves:** For each distinct location where unserved children are waiting (excluding the kitchen, as trays start there), a tray must be moved to that location *unless* a tray is already present there. Count the number of such locations. Each requires a `move_tray` action.
    5.  **Sum Components:** The total heuristic value is the sum of the estimated actions:
        - Number of `serve` actions (equal to the number of unserved children).
        - Number of `make` actions (estimated new sandwiches needed).
        - Number of `put_on_tray` actions (estimated new sandwiches needed).
        - Number of `move_tray` actions (estimated distinct locations needing a tray).

    This heuristic is non-admissible as it doesn't strictly track resource availability (like bread, content, notexist sandwich objects, or trays in the kitchen for `put_on_tray`) and assumes these resources will be available when needed for the estimated actions. However, it captures the main sequence of steps and bottlenecks.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Pre-processes static facts to map children to their waiting places.
        Identifies all children from the goal state.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Map children to their waiting places from static facts
        self.child_waiting_place = {}
        for fact in self.static_facts:
            if match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:3]
                self.child_waiting_place[child] = place

        # Identify all children from the goal facts (those who need to be served)
        self.all_children_in_goal = set()
        for goal in self.goals:
             if match(goal, "served", "*"):
                 child = get_parts(goal)[1]
                 self.all_children_in_goal.add(child)


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state
        total_cost = 0

        # 1. Identify Unserved Children
        served_children_in_state = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_children_in_state.add(get_parts(fact)[1])

        unserved_children = self.all_children_in_goal - served_children_in_state

        # If all children are served, the heuristic is 0
        if not unserved_children:
            return 0

        # Add cost for the final 'serve' action for each unserved child
        serve_actions = len(unserved_children)
        total_cost += serve_actions

        # 2. Estimate Sandwiches to Make
        # Count sandwiches already made (either in kitchen or on a tray)
        made_sandwiches = set()
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                made_sandwiches.add(get_parts(fact)[1])
            elif match(fact, "ontray", "*", "*"):
                 made_sandwiches.add(get_parts(fact)[1])

        made_sandwiches_count = len(made_sandwiches)

        # Number of new sandwiches needed is unserved children minus those already made
        new_sandwiches_needed = max(0, len(unserved_children) - made_sandwiches_count)

        # Add cost for making new sandwiches
        make_actions = new_sandwiches_needed
        total_cost += make_actions

        # 3. Estimate Sandwiches to Put on Tray
        # Each new sandwich needs to be put on a tray
        put_actions = new_sandwiches_needed
        total_cost += put_actions

        # 4. Estimate Tray Moves
        # Identify distinct non-kitchen places where unserved children are waiting
        target_places_needing_tray = set()
        for child in unserved_children:
            # Get the waiting place for this child from the pre-processed static facts
            place = self.child_waiting_place.get(child)
            if place and place != 'kitchen': # Only consider non-kitchen places
                target_places_needing_tray.add(place)

        # Identify places where trays are currently located in the state
        places_with_trays_in_state = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                 # Check if the first argument is a tray object
                 obj_name = get_parts(fact)[1]
                 # We need to parse the objects list from the problem file to know types reliably,
                 # but a simple check like 'startswith("tray")' is a reasonable heuristic approximation
                 # given the naming convention in the examples.
                 # A more robust way would involve parsing the :objects section.
                 # For this heuristic, let's assume objects starting with 'tray' are trays.
                 if obj_name.startswith("tray"):
                    places_with_trays_in_state.add(get_parts(fact)[2])

        # Count how many target places do NOT currently have a tray
        move_actions = len(target_places_needing_tray - places_with_trays_in_state)
        total_cost += move_actions

        # Return the total estimated cost
        return total_cost

