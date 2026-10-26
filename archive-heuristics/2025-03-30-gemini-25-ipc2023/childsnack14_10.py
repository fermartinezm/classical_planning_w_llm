import os
import sys
from fnmatch import fnmatch # Used in match function

# Make sure Heuristic base class is importable
# This setup tries to find the 'heuristics.heuristic_base' module
# assuming common project structures. Adjust if necessary for your environment.
try:
    # Assumes the script is run in a context where 'heuristics' package is available
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # If run directly or package not set up, try adjusting path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    # Retry import
    try:
        from heuristics.heuristic_base import Heuristic
    except ImportError:
        # Final attempt assuming script is in project root and heuristics is subdir
        if current_dir not in sys.path:
             sys.path.append(current_dir)
        try:
            from heuristics.heuristic_base import Heuristic
        except ImportError:
             # If still not found, raise the error.
             raise ImportError("Could not import Heuristic base class from heuristics.heuristic_base. "
                               "Please ensure the path is correct or the package is installed.")


def get_parts(fact):
    """
    Extract the components of a PDDL fact string.

    Removes the surrounding parentheses and splits the string by spaces.
    Example: "(at tray1 kitchen)" -> ["at", "tray1", "kitchen"]

    Args:
        fact (str): The PDDL fact string.

    Returns:
        list: A list of strings representing the parts of the fact.
              Returns an empty list if the fact format is invalid (e.g., missing parentheses).
    """
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Return empty list for non-strings or improperly formatted facts
        return []
    # Split the content inside the parentheses
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern using string equality or wildcards.

    Args:
        fact (str): The complete fact as a string, e.g., "(waiting child1 table1)".
        *args: A variable number of strings representing the pattern components.
               Use '*' as a wildcard to match any single component at that position.

    Returns:
        bool: True if the fact structure (number of parts) and components match the
              pattern (considering wildcards), False otherwise.
    """
    parts = get_parts(fact)
    # Check if the number of parts in the fact matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check each part against the corresponding pattern argument using fnmatch for wildcard support
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'childsnacks'.

    # Summary
    This heuristic estimates the remaining number of actions required to reach a
    goal state where all specified children are served. It calculates the cost by
    summing a base cost for each unserved child (representing making, putting on
    a tray, and serving a sandwich) and an estimated movement cost (representing
    moving trays to unique non-kitchen locations where children are waiting).

    # Assumptions
    - Each child specified in the goal requires exactly one sandwich to be served.
    - Sandwiches must be made in the 'kitchen' location using available bread and content.
    - Trays are required to transport sandwiches from the 'kitchen' to the child's location.
    - The heuristic simplifies several aspects for efficiency:
        - It assumes 3 core actions (make, put_on_tray, serve) are needed per unserved child.
        - It estimates tray movement cost as 1 action per unique non-kitchen destination
          that still has unserved children waiting there.
        - It does not model tray capacity limits, the need for return trips to the kitchen,
          optimal routing/scheduling of trays, or resource contention (e.g., limited
          ingredients, trays, or specific sandwich types like gluten-free).
        - It assumes sufficient ingredients and sandwich objects are available to make
          the required sandwiches.

    # Heuristic Initialization
    - The constructor (`__init__`) processes the task's goal conditions (`task.goals`)
      to identify the set of children that must be served (`self.goal_children`).
    - It parses the static facts (`task.static`) to build a mapping from each child
      to their fixed waiting location (`self.child_locations`). This map is used
      later to determine where unserved children are.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unserved Children:** In the `__call__` method, the current `state`
        (represented as a frozenset of fact strings) is examined. The set of children
        for whom the `(served ?child)` predicate is true in the state is determined.
        This set is compared against `self.goal_children` to find the set of
        `unserved_children`.
    2.  **Count Unserved:** The number of unserved children, `N_unserved`, is calculated.
        If `N_unserved` is 0, it implies all goal children are served, and the
        heuristic returns 0.
    3.  **Calculate Base Cost:** A base cost is estimated assuming 3 fundamental actions
        are required for each unserved child: making the sandwich, putting it on a tray,
        and serving it. The total base cost is `base_cost = 3 * N_unserved`.
    4.  **Calculate Movement Cost:** The heuristic determines the set of unique waiting
        locations for all `unserved_children` by looking them up in `self.child_locations`.
        This set of locations is then filtered to include only those that are *not* the
        'kitchen'. The number of such unique non-kitchen locations represents the estimated
        movement cost, `N_unique_locations_not_kitchen`. This approximates the minimum
        number of distinct non-kitchen places that must be visited by a tray at least once.
    5.  **Total Heuristic Value:** The final heuristic estimate is the sum of the base
        cost and the movement cost: `h = base_cost + N_unique_locations_not_kitchen`.
        This value estimates the remaining actions needed to reach the goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children and their locations.

        Args:
            task: The planning task object. Provides access to:
                  - `task.goals` (frozenset): Goal facts.
                  - `task.static` (frozenset): Static facts (true in all states).
        """
        # Initialize the base class if it has its own initialization logic
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # Identify the set of children that need to be served according to the goal facts
        self.goal_children = set()
        for goal_fact in self.goals:
            # Use the match helper function for safe parsing and pattern matching
            if match(goal_fact, "served", "*"):
                parts = get_parts(goal_fact)
                # Ensure the fact has the expected structure (predicate child)
                if len(parts) == 2:
                    child_name = parts[1]
                    self.goal_children.add(child_name)

        # Store the fixed waiting location for every child from the static facts
        self.child_locations = {}
        for static_fact in static_facts:
            # Use match to find (waiting ?child ?place) facts
            if match(static_fact, "waiting", "*", "*"):
                parts = get_parts(static_fact)
                # Ensure the fact has the expected structure (predicate child place)
                if len(parts) == 3:
                    child_name = parts[1]
                    location_name = parts[2]
                    self.child_locations[child_name] = location_name

        # Flag to ensure warning for missing location is printed only once per instance
        self._warned_missing_location = False

    def __call__(self, node):
        """
        Calculate the heuristic value (estimated cost to goal) for the given state node.

        Args:
            node: The state node in the search graph. Contains `node.state`
                  (a frozenset of strings representing the current world state).

        Returns:
            int: An estimated cost (number of actions) to reach a goal state.
                 Returns 0 if the current state satisfies the goal conditions
                 related to serving children.
        """
        state = node.state

        # Determine which of the goal children are already served in the current state
        served_children_in_state = set()
        for fact in state:
            # Use match to find (served ?child) facts
            if match(fact, "served", "*"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    child_name = parts[1]
                    # Check if this served child is one of the children targeted by the goal
                    if child_name in self.goal_children:
                        served_children_in_state.add(child_name)

        # Identify the set of children who still need to be served
        unserved_children = self.goal_children - served_children_in_state
        N_unserved = len(unserved_children)

        # If there are no unserved children among the goal children, the goal is considered reached
        # in terms of serving. Return 0 cost.
        if N_unserved == 0:
            return 0

        # Calculate the base cost: 3 actions (make, put_on_tray, serve) per unserved child.
        # This is a simplification, ignoring specific sandwich types and resource availability.
        base_cost = 3 * N_unserved

        # Calculate the movement cost: Estimate 1 'move_tray' action is needed for each
        # unique non-kitchen location where at least one unserved child is waiting.
        unique_non_kitchen_locations = set()
        for child in unserved_children:
            # Look up the child's waiting location using the map built during initialization
            if child in self.child_locations:
                location = self.child_locations[child]
                # If the location is not the central 'kitchen', add it to the set
                # of unique destinations that need to be visited.
                if location != 'kitchen':
                    unique_non_kitchen_locations.add(location)
            else:
                # Handle the unlikely case where a goal child's location is unknown.
                # This might indicate an error in the PDDL instance file.
                # Print a warning once to stderr if this occurs.
                if not self._warned_missing_location:
                    print(f"Warning: Heuristic could not find waiting location for goal child '{child}' "
                          "in static facts. This child will be ignored for movement cost calculation.",
                          file=sys.stderr)
                    self._warned_missing_location = True # Set flag to avoid repeated warnings

        # The movement cost is the number of unique non-kitchen locations identified.
        movement_cost = len(unique_non_kitchen_locations)

        # The total heuristic value is the sum of the base cost and the estimated movement cost.
        heuristic_value = base_cost + movement_cost

        # Return the calculated heuristic value. It must be a non-negative integer.
        return heuristic_value
