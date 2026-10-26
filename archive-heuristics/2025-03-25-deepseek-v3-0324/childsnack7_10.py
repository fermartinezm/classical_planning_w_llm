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
    2. Accounting for sandwiches that need to be made (considering gluten-free requirements)
    3. Considering tray movements needed to serve children at different locations
    4. Tracking sandwich preparation and placement on trays

    # Assumptions:
    - Each child needs exactly one sandwich
    - Gluten-allergic children must be served gluten-free sandwiches
    - Sandwiches can be made in any order as long as requirements are met
    - Trays can carry multiple sandwiches
    - Moving a tray between locations costs 1 action

    # Heuristic Initialization
    - Extract information about which children are allergic to gluten
    - Identify which bread and content portions are gluten-free
    - Note initial waiting locations of children

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (main component of heuristic)
    2. For each unserved child:
       a) If allergic to gluten:
          - Check if a gluten-free sandwich exists on a tray at their location
          - If not, check if one can be made from available gluten-free ingredients
          - Add costs for making and placing sandwich if needed
       b) If not allergic:
          - Check if any sandwich exists on a tray at their location
          - If not, check if one can be made from available ingredients
          - Add costs for making and placing sandwich if needed
    3. Account for tray movements:
       - If a tray needs to move to serve a child, add movement cost
    4. Consider sandwich preparation:
       - Each sandwich requires at least 1 action to make
       - Each sandwich requires 1 action to place on tray
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

        # Extract children's waiting locations
        self.child_locations = {}
        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                self.child_locations[parts[1]] = parts[2]

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Check if we're already in a goal state
        if self.goals <= state:
            return 0

        # Count unserved children
        served_children = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])
        
        unserved_children = (self.allergic_children | self.normal_children) - served_children
        total_cost = len(unserved_children) * 3  # Base cost: make, put on tray, serve

        # Track available sandwiches
        kitchen_sandwiches = set()
        gluten_free_sandwiches = set()
        sandwiches_on_trays = set()
        gluten_free_on_trays = set()
        tray_locations = {}  # tray -> location
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at_kitchen_sandwich", "*"):
                kitchen_sandwiches.add(parts[1])
            elif match(fact, "no_gluten_sandwich", "*"):
                gluten_free_sandwiches.add(parts[1])
            elif match(fact, "ontray", "*", "*"):
                sandwiches_on_trays.add(parts[1])
                if parts[1] in gluten_free_sandwiches:
                    gluten_free_on_trays.add(parts[1])
            elif match(fact, "at", "*", "*") and parts[1].startswith("tray"):
                tray_locations[parts[1]] = parts[2]

        # Check if we need to make more sandwiches
        needed_normal = 0
        needed_gluten_free = 0
        
        for child in unserved_children:
            if child in self.allergic_children:
                # Check if there's a gluten-free sandwich at their location
                found = False
                for sandwich in gluten_free_on_trays:
                    tray = None
                    for fact in state:
                        if match(fact, "ontray", sandwich, "*"):
                            tray = get_parts(fact)[2]
                            break
                    if tray and tray_locations.get(tray) == self.child_locations[child]:
                        found = True
                        break
                if not found:
                    needed_gluten_free += 1
            else:
                # Check if there's any sandwich at their location
                found = False
                for sandwich in sandwiches_on_trays:
                    tray = None
                    for fact in state:
                        if match(fact, "ontray", sandwich, "*"):
                            tray = get_parts(fact)[2]
                            break
                    if tray and tray_locations.get(tray) == self.child_locations[child]:
                        found = True
                        break
                if not found:
                    needed_normal += 1

        # Add costs for making sandwiches
        total_cost += needed_normal + needed_gluten_free  # make_sandwich actions
        total_cost += needed_normal + needed_gluten_free  # put_on_tray actions

        # Add potential tray movement costs
        unique_locations = {self.child_locations[child] for child in unserved_children}
        if len(unique_locations) > 1:
            total_cost += len(unique_locations) - 1  # approximate tray movements

        return total_cost
