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
    This heuristic estimates the number of actions needed to serve all children
    by considering the following components:
    1. Making sandwiches (regular and gluten-free)
    2. Putting sandwiches on trays
    3. Moving trays to children's locations
    4. Serving sandwiches to children

    # Assumptions:
    - Each child needs exactly one sandwich.
    - Gluten-allergic children must be served gluten-free sandwiches.
    - Regular children can be served any sandwich.
    - Multiple sandwiches can be placed on a single tray.
    - The kitchen is the only place where sandwiches can be made.

    # Heuristic Initialization
    - Extract information about which children are allergic to gluten.
    - Identify which bread and content portions are gluten-free.
    - Note the waiting locations of all children.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children and categorize them by gluten requirements.
    2. For gluten-allergic children:
       - Count available gluten-free sandwiches (on trays or in kitchen).
       - Count potential gluten-free sandwiches that can still be made.
       - Each missing sandwich requires a make_sandwich_no_gluten action.
    3. For regular children:
       - Count all available sandwiches (on trays or in kitchen).
       - Count potential sandwiches that can still be made.
       - Each missing sandwich requires a make_sandwich action.
    4. For each sandwich that needs to be served:
       - If not on a tray: add 1 for put_on_tray action.
       - If tray not at child's location: add 1 for move_tray action.
       - Add 1 for serve_sandwich(_no_gluten) action.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract information about children's allergies
        self.allergic_children = set()
        self.regular_children = set()
        for fact in self.static:
            if match(fact, "allergic_gluten", "*"):
                self.allergic_children.add(get_parts(fact)[1])
            elif match(fact, "not_allergic_gluten", "*"):
                self.regular_children.add(get_parts(fact)[1])

        # Extract gluten-free ingredients
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()
        for fact in self.static:
            if match(fact, "no_gluten_bread", "*"):
                self.gluten_free_breads.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_content", "*"):
                self.gluten_free_contents.add(get_parts(fact)[1])

        # Extract waiting locations of children
        self.child_locations = {}
        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                self.child_locations[parts[1]] = parts[2]

    def __call__(self, node):
        """Estimate the number of actions needed to serve all children."""
        state = node.state

        # Count served and unserved children
        served_children = set()
        unserved_allergic = set()
        unserved_regular = set()
        
        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])
        
        for child in self.allergic_children:
            if child not in served_children:
                unserved_allergic.add(child)
        
        for child in self.regular_children:
            if child not in served_children:
                unserved_regular.add(child)

        # If all children are served, heuristic is 0
        if not unserved_allergic and not unserved_regular:
            return 0

        # Count available sandwiches
        available_gluten_free = 0
        available_regular = 0
        
        # Sandwiches in kitchen
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                sandwich = get_parts(fact)[1]
                if match(fact, "no_gluten_sandwich", "*") or any(
                    f"(no_gluten_sandwich {sandwich})" in state
                    for f in state if match(f, "no_gluten_sandwich", "*")
                ):
                    available_gluten_free += 1
                else:
                    available_regular += 1
        
        # Sandwiches on trays
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                sandwich = get_parts(fact)[1]
                if match(fact, "no_gluten_sandwich", "*") or any(
                    f"(no_gluten_sandwich {sandwich})" in state
                    for f in state if match(f, "no_gluten_sandwich", "*")
                ):
                    available_gluten_free += 1
                else:
                    available_regular += 1

        # Count potential sandwiches that can still be made
        potential_gluten_free = 0
        potential_regular = 0
        
        # Count available gluten-free ingredients
        available_gf_breads = sum(
            1 for fact in state if match(fact, "at_kitchen_bread", "*") and 
            get_parts(fact)[1] in self.gluten_free_breads
        )
        available_gf_contents = sum(
            1 for fact in state if match(fact, "at_kitchen_content", "*") and 
            get_parts(fact)[1] in self.gluten_free_contents
        )
        potential_gluten_free = min(available_gf_breads, available_gf_contents)
        
        # Count all available ingredients
        available_breads = sum(
            1 for fact in state if match(fact, "at_kitchen_bread", "*")
        )
        available_contents = sum(
            1 for fact in state if match(fact, "at_kitchen_content", "*")
        )
        potential_regular = min(available_breads, available_contents) - potential_gluten_free

        # Calculate missing sandwiches
        missing_gluten_free = max(0, len(unserved_allergic) - available_gluten_free)
        missing_regular = max(0, len(unserved_regular) - available_regular - 
                             max(0, available_gluten_free - len(unserved_allergic)))

        # Actions needed to make missing sandwiches
        make_actions = missing_gluten_free + missing_regular

        # Actions needed to put sandwiches on trays
        put_on_tray_actions = min(
            len(unserved_allergic) + len(unserved_regular),
            available_gluten_free + available_regular + potential_gluten_free + potential_regular
        )

        # Actions needed to move trays (assuming optimal tray usage)
        tray_locations = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                tray, loc = get_parts(fact)[1], get_parts(fact)[2]
                if tray.startswith("tray"):
                    tray_locations.add((tray, loc))
        
        # For each child, check if their location has a tray
        needed_moves = 0
        for child in unserved_allergic.union(unserved_regular):
            child_loc = self.child_locations[child]
            has_tray = any(loc == child_loc for (tray, loc) in tray_locations)
            if not has_tray:
                needed_moves += 1

        # Serve actions (one per child)
        serve_actions = len(unserved_allergic) + len(unserved_regular)

        # Total heuristic estimate
        return make_actions + put_on_tray_actions + needed_moves + serve_actions
