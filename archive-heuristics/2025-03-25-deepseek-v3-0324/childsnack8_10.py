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
    3. Tracking sandwiches that need to be placed on trays
    4. Considering tray movements needed to serve children

    # Assumptions:
    - Each child needs exactly one sandwich.
    - Gluten-allergic children must be served gluten-free sandwiches.
    - Non-allergic children can be served any sandwich.
    - Sandwiches must be made before being placed on trays.
    - Trays must be moved to the correct location before serving.

    # Heuristic Initialization
    - Extract information about which children are allergic to gluten from static facts.
    - Identify waiting locations for each child from static facts.
    - Store goal conditions (which children need to be served).

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (from goal conditions not met in current state).
    2. For each unserved child:
       - If allergic to gluten:
         - Check if a suitable gluten-free sandwich exists on a tray at their location
         - If not, count actions needed: make sandwich, put on tray, move tray (if needed), serve
       - If not allergic:
         - Check if any sandwich exists on a tray at their location
         - If not, count actions needed: make sandwich, put on tray, move tray (if needed), serve
    3. For sandwiches already on trays but not at correct locations:
       - Count tray movement actions needed
    4. For sandwiches in kitchen not on trays:
       - Count actions needed to put them on trays
    5. For sandwiches not yet made:
       - Count actions needed to make them from available ingredients
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract allergic children from static facts
        self.allergic_children = {
            get_parts(fact)[1] 
            for fact in self.static 
            if match(fact, "allergic_gluten", "*")
        }
        
        # Extract waiting locations for children from static facts
        self.child_locations = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in self.static
            if match(fact, "waiting", "*", "*")
        }

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if we're already in a goal state
        if self.goals <= state:
            return 0
            
        # Initialize counters for different types of actions needed
        make_sandwich_actions = 0
        put_on_tray_actions = 0
        move_tray_actions = 0
        serve_actions = 0
        
        # Track which children still need to be served
        unserved_children = {
            get_parts(goal)[1]
            for goal in self.goals
            if match(goal, "served", "*") and f"(served {get_parts(goal)[1]})" not in state
        }
        
        # Track sandwiches that exist (have been made)
        existing_sandwiches = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "at_kitchen_sandwich", "*") or match(fact, "ontray", "*", "*")
        }
        
        # Track sandwiches on trays and their locations
        sandwiches_on_trays = {}
        tray_locations = {}
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                parts = get_parts(fact)
                sandwiches_on_trays[parts[1]] = parts[2]
            elif match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1].startswith("tray"):
                    tray_locations[parts[1]] = parts[2]
        
        # Track gluten-free sandwiches
        gluten_free_sandwiches = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "no_gluten_sandwich", "*")
        }
        
        # For each unserved child, determine what actions are needed
        for child in unserved_children:
            child_location = self.child_locations[child]
            served = False
            
            # Check if there's already a suitable sandwich on a tray at the child's location
            if child in self.allergic_children:
                # Need gluten-free sandwich
                for sandwich in gluten_free_sandwiches:
                    if sandwich in sandwiches_on_trays:
                        tray = sandwiches_on_trays[sandwich]
                        if tray_locations.get(tray) == child_location:
                            served = True
                            break
            else:
                # Can use any sandwich
                for sandwich in existing_sandwiches:
                    if sandwich in sandwiches_on_trays:
                        tray = sandwiches_on_trays[sandwich]
                        if tray_locations.get(tray) == child_location:
                            served = True
                            break
            
            if not served:
                # Need to make a sandwich if none exists
                if not existing_sandwiches:
                    make_sandwich_actions += 1
                    existing_sandwiches.add("new_sandwich")  # Placeholder
                
                # Need to put sandwich on tray if not already on one
                put_on_tray_actions += 1
                
                # Need to move tray if not at child's location
                # We'll assume at least one tray is available (from initial state)
                move_tray_actions += 1
                
                # Finally serve the sandwich
                serve_actions += 1
        
        # Count sandwiches in kitchen that aren't on trays yet
        sandwiches_in_kitchen = sum(
            1 for fact in state if match(fact, "at_kitchen_sandwich", "*")
        )
        put_on_tray_actions += sandwiches_in_kitchen
        
        # Count sandwiches that are on trays but not at correct locations
        # (This is already covered by move_tray_actions in the child loop)
        
        # Total heuristic is sum of all estimated actions
        return (
            make_sandwich_actions 
            + put_on_tray_actions 
            + move_tray_actions 
            + serve_actions
        )
