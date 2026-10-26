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


class ChildsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Childsnack domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children by:
    1. Counting unserved children
    2. Estimating sandwiches needed (considering gluten allergies)
    3. Tracking sandwich preparation steps
    4. Accounting for tray movements

    # Assumptions:
    - Each child needs exactly one sandwich
    - Gluten-allergic children must be served gluten-free sandwiches
    - Sandwiches must be made, placed on trays, and trays moved to children
    - Multiple sandwiches can be on one tray
    - Kitchen is the only starting location for sandwich preparation

    # Heuristic Initialization
    - Extract static information about children's allergies and waiting locations
    - Identify gluten-free bread and content portions
    - Store goal conditions (served children)

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (main driver of heuristic)
    2. For each unserved child:
       - If allergic: check if gluten-free sandwich exists or can be made
       - If not allergic: check if any sandwich exists or can be made
    3. Estimate sandwich preparation steps:
       - Making sandwiches (1 action per sandwich)
       - Putting sandwiches on trays (1 action per sandwich)
    4. Estimate tray movement:
       - Each tray needs to be moved to the child's location (1 action per unique location)
    5. Sum all estimated actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract information about children's allergies
        self.allergic_children = set()
        self.normal_children = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif parts[0] == "not_allergic_gluten":
                self.normal_children.add(parts[1])

        # Extract gluten-free ingredients
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "no_gluten_bread":
                self.gluten_free_breads.add(parts[1])
            elif parts[0] == "no_gluten_content":
                self.gluten_free_contents.add(parts[1])

        # Store goal children (those that need to be served)
        self.goal_children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "served":
                self.goal_children.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Count unserved goal children
        unserved_children = set(self.goal_children)
        served_children = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "served":
                served_children.add(parts[1])
        unserved_children -= served_children

        if not unserved_children:
            return 0  # Goal reached

        # Track available resources
        available_sandwiches = set()
        gluten_free_sandwiches = set()
        sandwiches_on_trays = set()
        trays_at_kitchen = set()
        trays_at_locations = set()
        bread_at_kitchen = set()
        content_at_kitchen = set()
        waiting_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at_kitchen_sandwich":
                available_sandwiches.add(parts[1])
            elif parts[0] == "no_gluten_sandwich":
                gluten_free_sandwiches.add(parts[1])
            elif parts[0] == "ontray":
                sandwiches_on_trays.add(parts[1])
            elif parts[0] == "at" and parts[2] == "kitchen":
                trays_at_kitchen.add(parts[1])
            elif parts[0] == "at" and parts[2] != "kitchen":
                trays_at_locations.add(parts[1])
            elif parts[0] == "at_kitchen_bread":
                bread_at_kitchen.add(parts[1])
            elif parts[0] == "at_kitchen_content":
                content_at_kitchen.add(parts[1])
            elif parts[0] == "waiting":
                waiting_locations.add(parts[2])

        # Initialize cost
        total_cost = 0

        # For each unserved child, estimate actions needed
        for child in unserved_children:
            is_allergic = child in self.allergic_children

            # Find a suitable sandwich
            if is_allergic:
                # Need gluten-free sandwich
                suitable_sandwiches = gluten_free_sandwiches & (available_sandwiches | sandwiches_on_trays)
                if not suitable_sandwiches:
                    # Need to make a new gluten-free sandwich
                    if self.gluten_free_breads & bread_at_kitchen and self.gluten_free_contents & content_at_kitchen:
                        total_cost += 1  # make_sandwich_no_gluten action
            else:
                # Can use any sandwich
                if not (available_sandwiches | sandwiches_on_trays):
                    # Need to make a new sandwich
                    if bread_at_kitchen and content_at_kitchen:
                        total_cost += 1  # make_sandwich action

            # Estimate tray operations
            if not sandwiches_on_trays:
                # Need to put sandwich on tray
                if trays_at_kitchen:
                    total_cost += 1  # put_on_tray action
                else:
                    # Need to move tray back to kitchen first
                    total_cost += 1  # move_tray action

            # Estimate serving action
            total_cost += 1  # serve_sandwich(_no_gluten) action

        # Estimate tray movements to child locations
        total_cost += len(waiting_locations)  # move_tray actions

        return total_cost
