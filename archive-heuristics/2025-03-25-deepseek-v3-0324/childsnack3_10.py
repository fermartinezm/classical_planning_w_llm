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
    A domain-dependent heuristic for the Childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children by:
    1. Counting unsatisfied children (not yet served)
    2. Estimating sandwich preparation needs (considering gluten allergies)
    3. Tracking tray movements required to serve sandwiches

    # Assumptions:
    - Each child needs exactly one sandwich
    - Gluten-allergic children must be served gluten-free sandwiches
    - Sandwiches can be made in any order as long as requirements are met
    - Trays can carry multiple sandwiches but must move to serve

    # Heuristic Initialization
    - Extract static information about children's allergies and waiting locations
    - Identify gluten-free bread and content portions from static facts
    - Store goal conditions (which children need to be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (main driver of heuristic value)
    2. For each unserved child:
       a) If allergic to gluten:
          - Check if a gluten-free sandwich exists on a tray at their location
          - If not, check if one can be made from available gluten-free ingredients
          - Add costs for making and moving sandwich if needed
       b) If not allergic:
          - Check if any sandwich exists on a tray at their location
          - If not, check if one can be made from available ingredients
          - Add costs for making and moving sandwich if needed
    3. Estimate tray movements needed to serve sandwiches at correct locations
    4. Sum all required actions to get total heuristic value
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

        # Map children to their waiting locations
        self.child_locations = {}
        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                self.child_locations[parts[1]] = parts[2]

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        total_cost = 0

        # Check which children still need to be served
        served_children = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])

        unserved_children = (self.allergic_children | self.normal_children) - served_children

        # Track sandwiches on trays at different locations
        sandwiches_on_trays = {}
        tray_locations = {}
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1], get_parts(fact)[2]
                sandwiches_on_trays[s] = t
            elif match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                if obj.startswith("tray"):
                    tray_locations[obj] = loc

        # Track available sandwiches in kitchen
        kitchen_sandwiches = set()
        gluten_free_sandwiches = set()
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                kitchen_sandwiches.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_sandwich", "*"):
                gluten_free_sandwiches.add(get_parts(fact)[1])

        # Track available ingredients
        available_breads = set()
        available_contents = set()
        for fact in state:
            if match(fact, "at_kitchen_bread", "*"):
                available_breads.add(get_parts(fact)[1])
            elif match(fact, "at_kitchen_content", "*"):
                available_contents.add(get_parts(fact)[1])

        for child in unserved_children:
            child_loc = self.child_locations[child]
            served = False

            # Check if there's already a suitable sandwich on a tray at the child's location
            for s, t in sandwiches_on_trays.items():
                if tray_locations.get(t, None) == child_loc:
                    if child in self.allergic_children:
                        if s in gluten_free_sandwiches:
                            served = True
                            break
                    else:
                        served = True
                        break

            if served:
                continue  # No additional actions needed for this child

            # Estimate actions needed to serve this child
            if child in self.allergic_children:
                # Need gluten-free sandwich
                # Check if one exists in kitchen
                gf_in_kitchen = gluten_free_sandwiches & kitchen_sandwiches
                if gf_in_kitchen:
                    # Put on tray and move (2 actions)
                    total_cost += 2
                else:
                    # Make new gluten-free sandwich (1), put on tray (1), move (1)
                    # Check if we have gluten-free ingredients
                    if self.gluten_free_breads & available_breads and self.gluten_free_contents & available_contents:
                        total_cost += 3
                    else:
                        # No gluten-free ingredients available - problem unsolvable
                        return float('inf')
            else:
                # Can use any sandwich
                if kitchen_sandwiches:
                    # Put on tray and move (2 actions)
                    total_cost += 2
                else:
                    # Make new sandwich (1), put on tray (1), move (1)
                    if available_breads and available_contents:
                        total_cost += 3
                    else:
                        # No ingredients available - problem unsolvable
                        return float('inf')

        return total_cost
