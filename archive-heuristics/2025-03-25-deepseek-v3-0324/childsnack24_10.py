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
    3. Tracking tray movements required to deliver sandwiches

    # Assumptions:
    - Each child needs exactly one sandwich
    - Gluten-allergic children must be served gluten-free sandwiches
    - Sandwiches must be made, placed on trays, and trays moved to children's locations
    - The kitchen is the central location where sandwiches are prepared

    # Heuristic Initialization
    - Extract static information about children's allergies and waiting locations
    - Identify gluten-free bread and content portions from static facts
    - Store goal conditions (which children need to be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (main component of heuristic)
    2. For each unserved child:
       - If allergic: check if suitable gluten-free sandwich exists or can be made
       - If not allergic: check if any sandwich exists or can be made
    3. Estimate sandwich preparation actions:
       - For needed sandwiches not yet made: 1 action per sandwich
       - For sandwiches not on trays: 1 action to put on tray
    4. Estimate tray movement actions:
       - For each unique location with waiting children: 1 action to move tray
    5. Sum all estimated actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract children's allergy information
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

        # Extract waiting locations for children
        self.child_locations = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "waiting":
                self.child_locations[parts[1]] = parts[2]

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic_value = 0

        # Identify served and unserved children
        served_children = set()
        unserved_children = set()
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "served":
                served_children.add(parts[1])
        
        # Check goal conditions to find all children that need to be served
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "served":
                child = parts[1]
                if child not in served_children:
                    unserved_children.add(child)

        # If all children are served, heuristic is 0
        if not unserved_children:
            return 0

        # Count sandwiches available in kitchen and on trays
        kitchen_sandwiches = set()
        gluten_free_sandwiches = set()
        sandwiches_on_trays = set()
        tray_locations = set()  # Locations where trays are placed
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at_kitchen_sandwich":
                kitchen_sandwiches.add(parts[1])
            elif parts[0] == "no_gluten_sandwich":
                gluten_free_sandwiches.add(parts[1])
            elif parts[0] == "ontray":
                sandwiches_on_trays.add(parts[1])
            elif parts[0] == "at" and parts[1].startswith("tray"):
                tray_locations.add(parts[2])

        # Count bread and content portions available in kitchen
        available_breads = set()
        available_contents = set()
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at_kitchen_bread":
                available_breads.add(parts[1])
            elif parts[0] == "at_kitchen_content":
                available_contents.add(parts[1])

        # For each unserved child, estimate required actions
        needed_gluten_free = 0
        needed_normal = 0
        locations_to_serve = set()

        for child in unserved_children:
            locations_to_serve.add(self.child_locations[child])
            
            if child in self.allergic_children:
                # Check if suitable sandwich exists or can be made
                found = False
                for s in gluten_free_sandwiches:
                    if s in kitchen_sandwiches or s in sandwiches_on_trays:
                        found = True
                        break
                
                if not found:
                    needed_gluten_free += 1
            else:
                # Check if any sandwich exists or can be made
                found = False
                for s in kitchen_sandwiches | sandwiches_on_trays:
                    found = True
                    break
                
                if not found:
                    needed_normal += 1

        # Estimate sandwich preparation actions
        # Each sandwich requires 1 make action + 1 put_on_tray action
        total_sandwiches_needed = needed_gluten_free + needed_normal
        heuristic_value += total_sandwiches_needed * 2

        # Estimate tray movement actions
        # Need to move trays to each unique location with waiting children
        # Subtract 1 if a tray is already at kitchen (starting point)
        trays_at_kitchen = sum(1 for fact in state if match(fact, "at", "tray*", "kitchen"))
        required_movements = max(0, len(locations_to_serve) - trays_at_kitchen)
        heuristic_value += required_movements

        # Each child needs 1 serve action
        heuristic_value += len(unserved_children)

        return heuristic_value
