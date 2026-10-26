from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return [] # Or raise an error, depending on expected input robustness

    # Remove parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the remaining effort to serve all children by summing up
    the estimated steps needed for each unserved child independently. It counts the
    number of "stages" a child is away from being served, considering the steps
    required to get a suitable sandwich to their location.

    # Assumptions
    - Each child in the goal must be served.
    - Serving a child requires a suitable sandwich on a tray at the child's waiting place.
    - A non-allergic child can eat any sandwich. An allergic child must eat a gluten-free sandwich.
    - The stages to serving a child are: Make Sandwich -> Sandwich At Kitchen -> Sandwich On Any Tray -> Sandwich On Tray At Child's Location -> Served.
    - The heuristic estimates the cost for each unserved child by counting how many of these stages are not yet met for the *cheapest* path to get a suitable sandwich to them, summing these counts. This ignores potential sharing of sandwiches or trays between children and is therefore inadmissible but potentially informative for greedy search.

    # Heuristic Initialization
    - Extracts the list of children who need to be served from the task goals.
    - Extracts static information about children's waiting places and allergy status
      from the task's static facts. This information is constant throughout the plan.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is calculated as the sum of estimated costs for each child
    that has not yet been served according to the current state.

    For each child `C` specified in the task goals:
    1.  Check if the predicate `(served C)` is present in the current state. If it is,
        this child is served, and contributes 0 to the heuristic.
    2.  If `(served C)` is *not* in the state (the child is unserved):
        a.  Add 1 to the child's cost (representing the final `serve` action).
        b.  Determine the child's waiting place `P` using the static `(waiting C P)` fact.
        c.  Check if there is *any* suitable sandwich `S` (gluten-free if `C` is allergic, any otherwise)
            that is currently on *any* tray `T` which is located `(at T P)`.
            -   Iterate through all facts `(ontray S T)` in the state.
            -   For each such sandwich `S` and tray `T`, check if `S` is suitable for `C`.
            -   If suitable, check if `(at T P)` is in the state.
            -   If such an `S` and `T` are found, the child is one stage away (needs only serving),
                and we proceed to the next child.
        d.  If no suitable sandwich is on a tray at the child's location `P`:
            -   Add 1 to the child's cost (representing the `move_tray` action needed to get a tray there).
            -   Check if there is *any* suitable sandwich `S` that is currently on *any* tray `T`
                (regardless of the tray's location).
                -   Iterate through all facts `(ontray S T)` in the state.
                -   For each such sandwich `S`, check if it is suitable for `C`.
                -   If such an `S` is found, the child is two stages away (needs moving tray and serving),
                    and we proceed to the next child.
            e.  If no suitable sandwich is on *any* tray:
                -   Add 1 to the child's cost (representing the `put_on_tray` action needed).
                -   Check if there is *any* suitable sandwich `S` that is currently `(at_kitchen_sandwich S)`.
                    -   Iterate through all facts `(at_kitchen_sandwich S)` in the state.
                    -   For each such sandwich `S`, check if it is suitable for `C`.
                    -   If such an `S` is found, the child is three stages away (needs put on tray, move tray, serve),
                        and we proceed to the next child.
            f.  If no suitable sandwich is `at_kitchen_sandwich`:
                -   Add 1 to the child's cost (representing the `make_sandwich` action needed).
                -   The child is four stages away (needs make, put on tray, move tray, serve). Proceed to the next child.

    3.  The total heuristic value is the sum of the costs calculated for each unserved child.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children, waiting places,
        and allergy status from the task definition.
        """
        self.goal_children = set()
        # Map child object name to their waiting place object name
        self.waiting_places = {}
        # Store allergy status for quick lookup
        self.allergic_children = {}
        self.not_allergic_children = {}

        # Extract goal children from goal facts
        for goal in task.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "served":
                self.goal_children.add(parts[1])

        # Extract static information from static facts
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: # Skip empty or malformed facts
                continue
            predicate = parts[0]
            if predicate == "waiting":
                child, place = parts[1:]
                self.waiting_places[child] = place
            elif predicate == "allergic_gluten":
                child = parts[1]
                self.allergic_children[child] = True
            elif predicate == "not_allergic_gluten":
                child = parts[1]
                self.not_allergic_children[child] = True

    def is_suitable(self, sandwich_s, child_c, state):
        """
        Checks if a given sandwich is suitable for a given child based on allergy status.

        Args:
            sandwich_s (str): The name of the sandwich object.
            child_c (str): The name of the child object.
            state (frozenset): The current state facts.

        Returns:
            bool: True if the sandwich is suitable, False otherwise.
        """
        is_gluten_free_s = "(no_gluten_sandwich " + sandwich_s + ")" in state
        is_allergic_c = self.allergic_children.get(child_c, False)
        # is_not_allergic_c = self.not_allergic_children.get(child_c, False) # Not strictly needed for logic

        if is_allergic_c:
            # Allergic children MUST have gluten-free sandwiches
            return is_gluten_free_s
        else:
            # Non-allergic children can have any sandwich (gluten or gluten-free)
            return True

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the state.

        Returns:
            int: The estimated cost to reach the goal state.
        """
        state = node.state
        total_cost = 0

        # Iterate through all children that need to be served according to the goal
        for child in self.goal_children:
            # Check if the child is already served in the current state
            if "(served " + child + ")" not in state:
                # This child is unserved, add cost
                child_cost = 0

                # Stage 5: Served (cost 1 for the serve action)
                child_cost += 1

                # Find the child's waiting place (guaranteed to exist by domain setup)
                waiting_place = self.waiting_places[child]

                # Check Stage 4: Suitable sandwich on a tray at the child's location
                suitable_on_tray_at_location = False
                for fact in state:
                    if match(fact, "ontray", "*", "*"):
                        s, t = get_parts(fact)[1:]
                        # Check if the sandwich is suitable for this child
                        if self.is_suitable(s, child, state):
                            # Check if the tray is at the child's waiting place
                            if "(at " + t + " " + waiting_place + ")" in state:
                                suitable_on_tray_at_location = True
                                break # Found a suitable sandwich on a tray at the location

                if not suitable_on_tray_at_location:
                    # Stage 3: Suitable sandwich on *any* tray (cost 1 for move_tray)
                    child_cost += 1

                    # Check Stage 3: Suitable sandwich on *any* tray
                    suitable_on_any_tray = False
                    for fact in state:
                        if match(fact, "ontray", "*", "*"):
                            s, t = get_parts(fact)[1:]
                            # Check if the sandwich is suitable for this child
                            if self.is_suitable(s, child, state):
                                suitable_on_any_tray = True
                                break # Found a suitable sandwich on any tray

                    if not suitable_on_any_tray:
                        # Stage 2: Suitable sandwich at the kitchen (cost 1 for put_on_tray)
                        child_cost += 1

                        # Check Stage 2: Suitable sandwich at the kitchen
                        suitable_at_kitchen = False
                        for fact in state:
                            if match(fact, "at_kitchen_sandwich", "*"):
                                s = get_parts(fact)[1]
                                # Check if the sandwich is suitable for this child
                                if self.is_suitable(s, child, state):
                                    suitable_at_kitchen = True
                                    break # Found a suitable sandwich at the kitchen

                        if not suitable_at_kitchen:
                            # Stage 1: Sandwich needs to be made (cost 1 for make_sandwich)
                            child_cost += 1

                # Add the calculated cost for this unserved child to the total
                total_cost += child_cost

        return total_cost

