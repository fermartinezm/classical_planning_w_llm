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
    2. Accounting for sandwiches that need to be made (considering gluten allergies)
    3. Considering tray movements needed to serve children at different locations
    4. Tracking sandwiches that are already on trays but not yet served

    # Assumptions:
    - Each child needs exactly one sandwich
    - Gluten-allergic children must be served gluten-free sandwiches
    - Sandwiches can be made in any order as long as requirements are met
    - Tray movements between locations are necessary when serving children at different places

    # Heuristic Initialization
    - Extract information about children's allergies from static facts
    - Identify waiting locations for each child from static facts
    - Track which bread and content portions are gluten-free

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (main component of heuristic)
    2. For each unserved child:
       a) If allergic to gluten:
          - Check if a gluten-free sandwich is available on a tray at their location
          - If not, check if one can be made from available gluten-free ingredients
       b) If not allergic:
          - Check if any sandwich is available on a tray at their location
          - If not, check if one can be made from available ingredients
    3. Account for tray movements needed to serve children at different locations
    4. Add costs for making sandwiches (1 action per sandwich)
    5. Add costs for putting sandwiches on trays (1 action per sandwich)
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

        # Extract waiting locations for each child
        self.child_locations = {}
        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                self.child_locations[parts[1]] = parts[2]

        # Track gluten-free ingredients
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

        # If all children are served, heuristic is 0
        if not unserved_children:
            return 0

        # Track available resources
        available_breads = set()
        available_contents = set()
        available_sandwiches = set()
        gluten_free_sandwiches = set()
        sandwiches_on_trays = set()
        tray_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at_kitchen_bread", "*"):
                available_breads.add(parts[1])
            elif match(fact, "at_kitchen_content", "*"):
                available_contents.add(parts[1])
            elif match(fact, "at_kitchen_sandwich", "*"):
                available_sandwiches.add(parts[1])
            elif match(fact, "no_gluten_sandwich", "*"):
                gluten_free_sandwiches.add(parts[1])
            elif match(fact, "ontray", "*", "*"):
                sandwiches_on_trays.add(parts[1])
            elif match(fact, "at", "*", "*") and parts[1].startswith("tray"):
                tray_locations[parts[1]] = parts[2]

        # For each unserved child, estimate actions needed
        for child in unserved_children:
            location = self.child_locations[child]
            is_allergic = child in self.allergic_children

            # Check if there's already a suitable sandwich on a tray at the child's location
            suitable_sandwich_found = False
            for tray, tray_loc in tray_locations.items():
                if tray_loc == location:
                    for sandwich in sandwiches_on_trays:
                        if (not is_allergic) or (sandwich in gluten_free_sandwiches):
                            suitable_sandwich_found = True
                            break
                    if suitable_sandwich_found:
                        break

            if suitable_sandwich_found:
                heuristic_value += 1  # serve_sandwich action
                continue

            # If no suitable sandwich on tray at location, we need to:
            # 1. Make a sandwich (if not already made)
            # 2. Put it on a tray
            # 3. Move tray to location (if not already there)
            # 4. Serve the sandwich

            # Check if we have a suitable sandwich in kitchen
            suitable_sandwich_available = False
            for sandwich in available_sandwiches:
                if (not is_allergic) or (sandwich in gluten_free_sandwiches):
                    suitable_sandwich_available = True
                    break

            if not suitable_sandwich_available:
                # Need to make a sandwich
                if is_allergic:
                    # Need gluten-free bread and content
                    if (self.gluten_free_breads & available_breads) and (
                        self.gluten_free_contents & available_contents
                    ):
                        heuristic_value += 1  # make_sandwich_no_gluten action
                    else:
                        # No suitable ingredients - problem likely unsolvable
                        return float("inf")
                else:
                    # Can use any bread and content
                    if available_breads and available_contents:
                        heuristic_value += 1  # make_sandwich action
                    else:
                        # No ingredients - problem likely unsolvable
                        return float("inf")

            # Need to put sandwich on tray (1 action)
            heuristic_value += 1

            # Need to move tray to location if not already there
            tray_at_location = any(
                loc == location for tray, loc in tray_locations.items()
            )
            if not tray_at_location:
                heuristic_value += 1  # move_tray action

            # Finally, serve the sandwich (1 action)
            heuristic_value += 1

        return heuristic_value
