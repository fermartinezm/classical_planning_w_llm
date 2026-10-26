from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
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
    # Basic check for number of parts vs args (handles fixed-arity predicates)
    if len(parts) != len(args) and '*' not in args:
         return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It sums up the estimated costs for four main stages:
    1. Serving the children.
    2. Making enough sandwiches.
    3. Putting sandwiches onto trays.
    4. Moving trays to the locations where children are waiting.

    # Assumptions
    - Each action (make_sandwich, put_on_tray, move_tray, serve_sandwich) costs 1.
    - Trays have unlimited capacity for sandwiches.
    - Resource availability (bread, content) for making sandwiches is not explicitly checked beyond the count of potential sandwiches.
    - The heuristic does not distinguish between gluten-free and regular sandwiches/resources when counting deficits, simplifying the calculation.
    - Any available tray can be used and moved to any required location.

    # Heuristic Initialization
    - Identifies all potential sandwich objects from the initial state (those initially marked with `notexist`).
    - Maps each child to their waiting place based on static facts.
    - Stores the set of goal facts (which specify which children need to be served).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1.  **Count Unserved Children:** Determine the number of children who are goal conditions (`(served ?c)`) but are not yet marked as served in the current state. This count represents the minimum number of `serve_sandwich` actions required. Add this count to the total heuristic value.
    2.  **Count Sandwiches to Make:** Determine the number of sandwiches that currently exist (i.e., are not marked with `notexist`). Compare this to the total number of unserved children. If more sandwiches are needed than currently exist, the difference is the estimated number of `make_sandwich` actions required. Add this difference (if positive) to the total heuristic value.
    3.  **Count Sandwiches to Put on Trays:** Count the number of sandwiches that are currently on trays (`ontray`). Compare this to the total number of unserved children. If more sandwiches need to be on trays than currently are, the difference is the estimated number of `put_on_tray` actions required. Add this difference (if positive) to the total heuristic value.
    4.  **Count Tray Movements:** Identify all distinct locations where unserved children are waiting. Count the number of trays that are currently located at any of these waiting places. If the number of distinct waiting locations is greater than the number of trays already at those locations, the difference is the estimated minimum number of `move_tray` actions required to get trays to the necessary places. Add this difference (if positive) to the total heuristic value.
    5.  **Sum Costs:** The total heuristic value is the sum of the costs calculated in steps 1, 2, 3, and 4.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Potential sandwich objects.
        - Child waiting places.
        - Goal facts.
        """
        self.goals = task.goals  # Goal conditions.
        self.static = task.static # Static facts.
        self.initial_state = task.initial_state # Initial state facts.

        # Identify all potential sandwich objects from the initial state (those initially marked with notexist)
        self.potential_sandwiches = {
            get_parts(fact)[1]
            for fact in self.initial_state
            if match(fact, "notexist", "*")
        }

        # Map children to their waiting places based on static facts.
        self.child_waiting_place = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in self.static
            if match(fact, "waiting", "*", "*")
        }

        # Identify all children that are goals (need to be served)
        self.children_to_serve = {
            get_parts(goal)[1]
            for goal in self.goals
            if match(goal, "served", "*")
        }


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        total_cost = 0  # Initialize action cost counter.

        # 1. Count Unserved Children
        served_children_in_state = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "served", "*")
        }
        not_served_children = self.children_to_serve - served_children_in_state
        num_not_served = len(not_served_children)

        # If all children are served, the heuristic is 0.
        if num_not_served == 0:
            return 0

        # Base cost is the number of serve actions needed
        total_cost += num_not_served

        # 2. Count Sandwiches to Make
        # Identify sandwiches that currently exist (are not marked with notexist)
        existing_sandwiches = {
            s for s in self.potential_sandwiches
            if f"(notexist {s})" not in state
        }
        num_existing_sandwiches = len(existing_sandwiches)

        # Need at least num_not_served sandwiches in total.
        # If we don't have enough existing sandwiches, we need to make more.
        sandwiches_to_make = max(0, num_not_served - num_existing_sandwiches)
        total_cost += sandwiches_to_make

        # 3. Count Sandwiches to Put on Trays
        # Count existing sandwiches that are currently on trays
        # More efficient way: build set of sandwiches on trays first
        sandwiches_on_tray_set = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "ontray", "*", "*")
        }
        sandwiches_ontray = len(existing_sandwiches.intersection(sandwiches_on_tray_set))


        # Need num_not_served sandwiches to be on trays eventually.
        # If not enough are currently on trays, we need to put more on.
        sandwiches_that_need_put_on_tray = max(0, num_not_served - sandwiches_ontray)
        total_cost += sandwiches_that_need_put_on_tray

        # 4. Count Tray Movements
        # Identify distinct locations where unserved children are waiting
        waiting_locations = {
            self.child_waiting_place[c]
            for c in not_served_children
            if c in self.child_waiting_place # Defensive check
        }
        num_waiting_locations = len(waiting_locations)

        # Count trays currently located at any of these waiting places
        trays_at_waiting_locations = sum(
            1 for fact in state
            if match(fact, "at", "*", "*") and get_parts(fact)[2] in waiting_locations
        )

        # Need at least one tray at each waiting location.
        # If not enough trays are already at waiting locations, we need to move more.
        tray_movements_needed = max(0, num_waiting_locations - trays_at_waiting_locations)
        total_cost += tray_movements_needed

        return total_cost
