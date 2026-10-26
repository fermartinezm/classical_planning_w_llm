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
    3. Accounting for sandwich preparation and tray movements

    # Assumptions:
    - Each child needs exactly one sandwich
    - Gluten-allergic children need gluten-free sandwiches
    - Sandwiches can be made in parallel if ingredients are available
    - Tray movements are needed when serving at different locations

    # Heuristic Initialization
    - Extract static information about children's allergies and waiting locations
    - Identify gluten-free bread and content portions
    - Store goal conditions (served children)

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (main driver of heuristic value)
    2. For each unserved child:
       - If allergic: check if gluten-free sandwich is available or can be made
       - If not allergic: check if any sandwich is available or can be made
    3. Estimate sandwich preparation actions:
       - Each sandwich requires 1 make_sandwich action
       - Gluten-free sandwiches require specific ingredients
    4. Estimate tray movements:
       - Each unique serving location may require a tray movement
    5. Combine counts with appropriate weights
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_locations = {}  # child -> place
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif parts[0] == "not_allergic_gluten":
                self.not_allergic_children.add(parts[1])
            elif parts[0] == "waiting":
                self.waiting_locations[parts[1]] = parts[2]
            elif parts[0] == "no_gluten_bread":
                self.no_gluten_breads.add(parts[1])
            elif parts[0] == "no_gluten_content":
                self.no_gluten_contents.add(parts[1])

        # Store goal children (those that need to be served)
        self.goal_children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "served":
                self.goal_children.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Count unserved children that are part of the goal
        unserved_children = 0
        allergic_unserved = 0
        for child in self.goal_children:
            if f"(served {child})" not in state:
                unserved_children += 1
                if child in self.allergic_children:
                    allergic_unserved += 1

        # If all children are served, return 0
        if unserved_children == 0:
            return 0

        # Count available resources
        available_breads = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*"))
        available_contents = sum(1 for fact in state if match(fact, "at_kitchen_content", "*"))
        available_sandwiches = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*"))
        gluten_free_sandwiches = sum(1 for fact in state if match(fact, "no_gluten_sandwich", "*"))
        sandwiches_on_trays = sum(1 for fact in state if match(fact, "ontray", "*", "*"))

        # Count available gluten-free resources
        available_gluten_free_breads = sum(
            1 for fact in state if match(fact, "at_kitchen_bread", "*") and 
            get_parts(fact)[1] in self.no_gluten_breads
        )
        available_gluten_free_contents = sum(
            1 for fact in state if match(fact, "at_kitchen_content", "*") and 
            get_parts(fact)[1] in self.no_gluten_contents
        )

        # Calculate sandwich needs
        regular_needed = max(0, unserved_children - allergic_unserved)
        gluten_free_needed = allergic_unserved

        # Estimate sandwich preparation actions
        make_actions = 0
        # Gluten-free sandwiches
        gf_sandwiches_available = gluten_free_sandwiches + any(
            match(fact, "no_gluten_sandwich", "*") for fact in state if match(fact, "ontray", "*", "*")
        )
        gf_deficit = max(0, gluten_free_needed - gf_sandwiches_available)
        gf_possible = min(available_gluten_free_breads, available_gluten_free_contents)
        make_actions += min(gf_deficit, gf_possible)

        # Regular sandwiches
        regular_sandwiches_available = (available_sandwiches + sandwiches_on_trays) - gf_sandwiches_available
        regular_deficit = max(0, regular_needed - regular_sandwiches_available)
        regular_possible = min(available_breads - available_gluten_free_breads, 
                              available_contents - available_gluten_free_contents)
        make_actions += min(regular_deficit, regular_possible)

        # Estimate tray movements
        # Count unique locations where unserved children are waiting
        unique_locations = set()
        for child in self.goal_children:
            if f"(served {child})" not in state and child in self.waiting_locations:
                unique_locations.add(self.waiting_locations[child])
        
        # Subtract kitchen (starting point) and trays already at locations
        trays_at_locations = set()
        for fact in state:
            if match(fact, "at", "*", "*") and not match(fact, "at", "*", "kitchen"):
                trays_at_locations.add(get_parts(fact)[2])
        
        move_actions = max(0, len(unique_locations - trays_at_locations))

        # Serving actions (one per child)
        serve_actions = unserved_children

        # Put on tray actions (one per sandwich)
        put_on_tray_actions = max(0, unserved_children - sandwiches_on_trays)

        # Weighted sum of all estimated actions
        # (make_sandwich is most expensive as it requires ingredients)
        return (make_actions * 2 + move_actions + serve_actions + put_on_tray_actions)
