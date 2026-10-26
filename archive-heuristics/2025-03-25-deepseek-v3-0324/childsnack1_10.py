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
    2. Accounting for sandwiches that need to be made (regular or gluten-free)
    3. Considering tray movements and sandwich placements
    4. Tracking existing sandwiches that can be served immediately

    # Assumptions:
    - Making a sandwich takes 1 action
    - Putting a sandwich on a tray takes 1 action
    - Serving a sandwich takes 1 action
    - Moving a tray between locations takes 1 action
    - Gluten-free sandwiches must be made separately with appropriate ingredients
    - We can optimize by serving multiple children at the same location with one tray movement

    # Heuristic Initialization
    - Extract information about children's allergies and waiting locations
    - Identify gluten-free bread and content portions
    - Note initial positions of trays

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (from goal conditions)
    2. For each unserved child:
       a. If allergic to gluten:
          - Check if a gluten-free sandwich exists on a tray at their location
          - If not, check if one can be made from available ingredients
       b. If not allergic:
          - Check if any sandwich exists on a tray at their location
          - If not, check if one can be made from available ingredients
    3. Estimate actions needed:
       - Each sandwich requires at least 1 action to make
       - Each sandwich needs 1 action to put on tray
       - Each unique tray movement to a location counts as 1 action
       - Each serving is 1 action
    4. Optimize by:
       - Counting sandwiches that can be served without making new ones
       - Grouping children at same location to minimize tray movements
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
        
        # Extract waiting locations for children
        self.child_locations = {}
        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                self.child_locations[parts[1]] = parts[2]

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Count unserved children (those not in goals)
        unserved_children = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                child = get_parts(goal)[1]
                if f"(served {child})" not in state:
                    unserved_children.add(child)
        
        if not unserved_children:
            return 0  # Goal state
        
        # Track available resources
        available_breads = set()
        available_contents = set()
        available_sandwiches = set()
        gluten_free_sandwiches = set()
        sandwiches_on_trays = {}  # {tray: {sandwich: location}}
        tray_locations = {}  # {tray: location}
        
        for fact in state:
            if match(fact, "at_kitchen_bread", "*"):
                available_breads.add(get_parts(fact)[1])
            elif match(fact, "at_kitchen_content", "*"):
                available_contents.add(get_parts(fact)[1])
            elif match(fact, "at_kitchen_sandwich", "*"):
                available_sandwiches.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_sandwich", "*"):
                gluten_free_sandwiches.add(get_parts(fact)[1])
            elif match(fact, "ontray", "*", "*"):
                parts = get_parts(fact)
                if parts[2] not in sandwiches_on_trays:
                    sandwiches_on_trays[parts[2]] = {}
                sandwiches_on_trays[parts[2]][parts[1]] = None
            elif match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1].startswith("tray"):
                    tray_locations[parts[1]] = parts[2]
        
        # Assign locations to sandwiches on trays
        for tray in sandwiches_on_trays:
            if tray in tray_locations:
                for sandwich in sandwiches_on_trays[tray]:
                    sandwiches_on_trays[tray][sandwich] = tray_locations[tray]
        
        # Initialize action count
        total_actions = 0
        
        # Track sandwiches we plan to make
        sandwiches_to_make = set()
        gluten_free_to_make = set()
        
        # Track tray movements needed
        tray_movements = set()
        
        for child in unserved_children:
            location = self.child_locations[child]
            served = False
            
            # Check if child is allergic
            is_allergic = child in self.allergic_children
            
            # First check sandwiches already on trays at the location
            for tray in sandwiches_on_trays:
                for sandwich in sandwiches_on_trays[tray]:
                    if sandwiches_on_trays[tray][sandwich] == location:
                        if is_allergic and sandwich in gluten_free_sandwiches:
                            total_actions += 1  # serve action
                            served = True
                            break
                        elif not is_allergic:
                            total_actions += 1  # serve action
                            served = True
                            break
                if served:
                    break
            
            if served:
                continue
            
            # If not served, we need to make a sandwich and bring it
            if is_allergic:
                # Need gluten-free sandwich
                if available_sandwiches & gluten_free_sandwiches:
                    # Can use existing gluten-free sandwich
                    pass
                else:
                    # Need to make one
                    gluten_free_to_make.add(child)
            else:
                # Can use any sandwich
                if available_sandwiches:
                    # Can use existing sandwich
                    pass
                else:
                    # Need to make one
                    sandwiches_to_make.add(child)
            
            # Need to move a tray to the location if not already there
            tray_at_location = any(
                tray in tray_locations and tray_locations[tray] == location 
                for tray in tray_locations
            )
            if not tray_at_location:
                tray_movements.add(location)
        
        # Add actions for making sandwiches
        total_actions += len(sandwiches_to_make) + len(gluten_free_to_make)
        
        # Add actions for putting sandwiches on trays (1 per sandwich)
        total_actions += len(sandwiches_to_make) + len(gluten_free_to_make)
        
        # Add actions for tray movements
        total_actions += len(tray_movements)
        
        # Add actions for serving (already counted during child iteration)
        
        return total_actions
