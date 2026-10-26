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
    2. Estimating the work needed to create sandwiches (regular and gluten-free)
    3. Accounting for tray movements and sandwich placements

    # Assumptions:
    - Each child needs exactly one sandwich
    - Gluten-allergic children must be served gluten-free sandwiches
    - Regular children can be served any sandwich
    - Multiple sandwiches can be placed on a single tray
    - Trays may need to be moved between locations

    # Heuristic Initialization
    - Extract information about children's allergies and waiting locations
    - Identify gluten-free bread and content portions
    - Store goal conditions (which children need to be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (both allergic and non-allergic)
    2. For allergic children:
       - Check if suitable gluten-free sandwiches exist or can be made
       - Count missing ingredients for gluten-free sandwiches
    3. For regular children:
       - Check if any sandwiches exist or can be made
       - Count missing ingredients for regular sandwiches
    4. Estimate tray movements:
       - Each tray needs to be moved to serve children at different locations
    5. Combine these estimates with appropriate weights
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract information about children's allergies
        self.allergic_children = {
            get_parts(fact)[1] for fact in self.static if match(fact, "allergic_gluten", "*")
        }
        self.non_allergic_children = {
            get_parts(fact)[1] for fact in self.static if match(fact, "not_allergic_gluten", "*")
        }

        # Extract gluten-free ingredients
        self.gluten_free_breads = {
            get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_bread", "*")
        }
        self.gluten_free_contents = {
            get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_content", "*")
        }

        # Extract waiting locations for children
        self.child_locations = {}
        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                self.child_locations[parts[1]] = parts[2]

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Count served and unserved children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_allergic = self.allergic_children - served_children
        unserved_regular = self.non_allergic_children - served_children
        total_unserved = len(unserved_allergic) + len(unserved_regular)

        if total_unserved == 0:
            return 0  # Goal state reached

        # Count available resources in current state
        kitchen_breads = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        kitchen_contents = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}
        kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        tray_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "ontray", "*", "*")}
        gluten_free_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        tray_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "tray*", "*")}

        # Calculate needed sandwiches
        needed_gluten_free = len(unserved_allergic)
        needed_regular = len(unserved_regular)

        # Count available gluten-free sandwiches (both in kitchen and on trays)
        available_gluten_free = len(gluten_free_sandwiches & (kitchen_sandwiches | tray_sandwiches))

        # Count available regular sandwiches (all sandwiches minus gluten-free ones)
        available_regular = len(kitchen_sandwiches | tray_sandwiches) - available_gluten_free

        # Calculate missing sandwiches
        missing_gluten_free = max(0, needed_gluten_free - available_gluten_free)
        missing_regular = max(0, needed_regular - available_regular)

        # Calculate needed ingredients for missing sandwiches
        needed_breads = missing_gluten_free + missing_regular
        available_breads = len(kitchen_breads)
        missing_breads = max(0, needed_breads - available_breads)

        needed_contents = missing_gluten_free + missing_regular
        available_contents = len(kitchen_contents)
        missing_contents = max(0, needed_contents - available_contents)

        # Calculate gluten-free ingredient constraints
        available_gluten_free_breads = len(kitchen_breads & self.gluten_free_breads)
        available_gluten_free_contents = len(kitchen_contents & self.gluten_free_contents)

        # If we're short on gluten-free ingredients, we need to account for that
        gluten_free_ingredient_shortage = max(
            0,
            missing_gluten_free - min(available_gluten_free_breads, available_gluten_free_contents)
        )

        # Estimate tray movements needed
        # We need to move trays to locations where unserved children are waiting
        needed_locations = set()
        for child in unserved_allergic | unserved_regular:
            needed_locations.add(self.child_locations[child])

        # Count how many of these locations already have trays
        served_locations = set(tray_locations.values())
        missing_locations = needed_locations - served_locations

        # Calculate heuristic value
        heuristic_value = (
            # Base cost for each unserved child
            total_unserved +
            # Cost for missing sandwiches (make_sandwich actions)
            (missing_gluten_free + missing_regular) +
            # Cost for missing ingredients (if any)
            (missing_breads + missing_contents) +
            # Additional cost if we're short on gluten-free ingredients
            gluten_free_ingredient_shortage * 2 +
            # Cost for tray movements
            len(missing_locations) * 2
        )

        return heuristic_value
