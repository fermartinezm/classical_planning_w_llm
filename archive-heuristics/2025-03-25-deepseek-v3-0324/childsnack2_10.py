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
    by considering:
    - Number of unserved children
    - Whether sandwiches need to be made
    - Whether sandwiches need to be put on trays
    - Whether trays need to be moved
    - Special handling for gluten-free sandwiches

    # Assumptions:
    - Making a sandwich takes 1 action
    - Putting a sandwich on a tray takes 1 action
    - Moving a tray takes 1 action per location change
    - Serving a sandwich takes 1 action
    - Gluten-free sandwiches require special handling

    # Heuristic Initialization
    - Extract information about which children are allergic to gluten
    - Extract waiting locations of children
    - Extract goal conditions (which children need to be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (main driver of heuristic value)
    2. For each unserved child:
       a. If allergic to gluten:
          - Check if a gluten-free sandwich exists on a tray at their location
          - If not, check if one can be made from available ingredients
       b. If not allergic:
          - Check if any sandwich exists on a tray at their location
          - If not, check if one can be made from available ingredients
    3. Estimate actions needed:
       - 1 action per sandwich creation
       - 1 action per tray loading
       - 1 action per tray movement (if needed)
       - 1 action per serving
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
        
        # Extract waiting locations
        self.child_locations = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "waiting":
                self.child_locations[parts[1]] = parts[2]

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if goal is already reached
        if self.goals <= state:
            return 0
            
        # Count unserved children in goals
        unserved_children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "served":
                child = parts[1]
                if f"(served {child})" not in state:
                    unserved_children.add(child)
        
        total_cost = 0
        
        # Track available resources
        available_bread = set()
        available_content = set()
        available_gluten_free_bread = set()
        available_gluten_free_content = set()
        available_sandwiches = set()
        available_gluten_free_sandwiches = set()
        sandwiches_on_trays = set()
        tray_locations = {}
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at_kitchen_bread":
                available_bread.add(parts[1])
            elif parts[0] == "at_kitchen_content":
                available_content.add(parts[1])
            elif parts[0] == "no_gluten_bread":
                available_gluten_free_bread.add(parts[1])
            elif parts[0] == "no_gluten_content":
                available_gluten_free_content.add(parts[1])
            elif parts[0] == "at_kitchen_sandwich":
                available_sandwiches.add(parts[1])
            elif parts[0] == "no_gluten_sandwich":
                available_gluten_free_sandwiches.add(parts[1])
            elif parts[0] == "ontray":
                sandwiches_on_trays.add(parts[1])
            elif parts[0] == "at" and parts[1].startswith("tray"):
                tray_locations[parts[1]] = parts[2]
        
        for child in unserved_children:
            location = self.child_locations[child]
            is_allergic = child in self.allergic_children
            
            # Check if there's already a suitable sandwich on a tray at the location
            found_sandwich = False
            for tray, tray_loc in tray_locations.items():
                if tray_loc == location:
                    for fact in state:
                        parts = get_parts(fact)
                        if parts[0] == "ontray" and parts[2] == tray:
                            sandwich = parts[1]
                            if (not is_allergic) or (f"(no_gluten_sandwich {sandwich})" in state):
                                found_sandwich = True
                                break
                    if found_sandwich:
                        break
            
            if found_sandwich:
                total_cost += 1  # serve action
                continue
            
            # Need to make and deliver a sandwich
            total_cost += 1  # make sandwich
            
            # Check if we need to put it on a tray
            if available_sandwiches or available_gluten_free_sandwiches:
                total_cost += 1  # put on tray
            
            # Check if we need to move the tray
            kitchen_trays = [tray for tray, loc in tray_locations.items() if loc == "kitchen"]
            if not kitchen_trays:
                # Need to move a tray back to kitchen first
                total_cost += 1  # move tray to kitchen
            
            # Move tray to child's location if not already there
            if location != "kitchen":
                total_cost += 1  # move tray to location
            
            total_cost += 1  # serve action
        
        return total_cost
