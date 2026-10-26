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
    1. Counting unsatisfied children (not yet served)
    2. Estimating sandwich preparation needs (considering gluten allergies)
    3. Accounting for tray movements and sandwich placements

    # Assumptions:
    - Each child needs exactly one sandwich
    - Gluten-allergic children must be served gluten-free sandwiches
    - Sandwiches can be made in parallel if ingredients are available
    - Tray movements are needed when serving at different locations

    # Heuristic Initialization
    - Extract static information about children's allergies and waiting locations
    - Identify gluten-free bread and content portions
    - Store goal conditions (which children need to be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (main driver of heuristic value)
    2. For each unserved child:
       - If allergic: check if suitable gluten-free sandwich exists or can be made
       - If not allergic: check if any sandwich exists or can be made
    3. Estimate sandwich preparation actions:
       - Each sandwich requires 1 make_sandwich action
       - Gluten-free sandwiches require specific ingredients
    4. Estimate tray operations:
       - Each sandwich needs to be placed on a tray (1 action)
       - Trays may need to be moved to serving locations
    5. Serving actions:
       - Each child needs 1 serve action
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract information about children's allergies
        self.allergic_children = set()
        self.normal_children = set()
        for fact in self.static:
            if match(fact, "allergic_gluten", "*"):
                self.allergic_children.add(get_parts(fact)[1])
            elif match(fact, "not_allergic_gluten", "*"):
                self.normal_children.add(get_parts(fact)[1])

        # Extract gluten-free ingredients
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()
        for fact in self.static:
            if match(fact, "no_gluten_bread", "*"):
                self.gluten_free_breads.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_content", "*"):
                self.gluten_free_contents.add(get_parts(fact)[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic_value = 0

        # Count unserved children
        unserved_children = set()
        for child in self.allergic_children | self.normal_children:
            if f"(served {child})" not in state:
                unserved_children.add(child)

        if not unserved_children:
            return 0  # Goal reached

        # Analyze current sandwiches
        kitchen_sandwiches = set()
        tray_sandwiches = set()
        gluten_free_sandwiches = set()
        
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                kitchen_sandwiches.add(get_parts(fact)[1])
            elif match(fact, "ontray", "*", "*"):
                tray_sandwiches.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_sandwich", "*"):
                gluten_free_sandwiches.add(get_parts(fact)[1])

        # Count available bread and content in kitchen
        available_breads = set()
        available_contents = set()
        for fact in state:
            if match(fact, "at_kitchen_bread", "*"):
                available_breads.add(get_parts(fact)[1])
            elif match(fact, "at_kitchen_content", "*"):
                available_contents.add(get_parts(fact)[1])

        # Estimate sandwich preparation needs
        needed_sandwiches = len(unserved_children)
        needed_gluten_free = len(unserved_children & self.allergic_children)
        
        # Available gluten-free sandwiches (on tray or in kitchen)
        available_gluten_free = len(gluten_free_sandwiches & (kitchen_sandwiches | tray_sandwiches))
        
        # Additional gluten-free sandwiches needed
        additional_gluten_free = max(0, needed_gluten_free - available_gluten_free)
        
        # Check if we have enough gluten-free ingredients
        available_gluten_free_pairs = min(
            len(available_breads & self.gluten_free_breads),
            len(available_contents & self.gluten_free_contents)
        )
        
        # Each missing gluten-free sandwich requires preparation
        heuristic_value += min(additional_gluten_free, available_gluten_free_pairs)
        
        # Regular sandwiches needed (total - gluten-free)
        regular_needed = needed_sandwiches - needed_gluten_free
        available_regular = len((kitchen_sandwiches | tray_sandwiches) - gluten_free_sandwiches)
        additional_regular = max(0, regular_needed - available_regular)
        
        # Check available regular ingredients
        available_regular_pairs = min(
            len(available_breads - self.gluten_free_breads),
            len(available_contents - self.gluten_free_contents)
        )
        
        # Each missing regular sandwich requires preparation
        heuristic_value += min(additional_regular, available_regular_pairs)

        # Estimate tray operations
        # Each sandwich needs to be put on tray (if not already)
        sandwiches_not_on_tray = kitchen_sandwiches
        heuristic_value += len(sandwiches_not_on_tray)

        # Estimate tray movements
        # We'll need at least one tray movement per unique serving location
        serving_locations = set()
        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                if parts[1] in unserved_children:
                    serving_locations.add(parts[2])
        
        # Check if trays are already at needed locations
        trays_at_needed_locations = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[2] in serving_locations:
                    trays_at_needed_locations.add(parts[2])
        
        # Additional tray movements needed
        heuristic_value += max(0, len(serving_locations) - len(trays_at_needed_locations))

        # Each child needs to be served (1 action per child)
        heuristic_value += len(unserved_children)

        return heuristic_value
