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
    2. Estimating sandwiches needed (considering gluten allergies)
    3. Accounting for sandwich preparation and tray movements

    # Assumptions:
    - Each child needs exactly one sandwich
    - Gluten-allergic children must be served gluten-free sandwiches
    - Non-allergic children can be served any sandwich
    - Multiple sandwiches can be placed on a single tray
    - All bread and content portions start in the kitchen

    # Heuristic Initialization
    - Extract allergic children from static facts
    - Extract goal conditions (served children)
    - Identify gluten-free bread and content portions

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (total children minus already served)
    2. For each unserved child:
       - If allergic and no suitable sandwich exists: 
         * Need to make gluten-free sandwich (1 action)
         * Put on tray (1 action)
         * Serve (1 action)
       - If not allergic and no sandwich exists:
         * Make sandwich (1 action)
         * Put on tray (1 action)
         * Serve (1 action)
       - If suitable sandwich exists on tray at correct location:
         * Just serve (1 action)
       - If suitable sandwich exists but not on tray:
         * Put on tray (1 action)
         * Serve (1 action)
    3. Estimate tray movements needed (1 per unique location)
    4. Sum all required actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract allergic children
        self.allergic_children = {
            get_parts(fact)[1] 
            for fact in self.static 
            if match(fact, "allergic_gluten", "*")
        }
        
        # Extract gluten-free ingredients
        self.gluten_free_breads = {
            get_parts(fact)[1] 
            for fact in self.static 
            if match(fact, "no_gluten_bread", "*")
        }
        self.gluten_free_contents = {
            get_parts(fact)[1] 
            for fact in self.static 
            if match(fact, "no_gluten_content", "*")
        }

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Count served and waiting children
        served_children = {
            get_parts(fact)[1] 
            for fact in state 
            if match(fact, "served", "*")
        }
        waiting_children = {
            (get_parts(fact)[1], get_parts(fact)[2]) 
            for fact in state 
            if match(fact, "waiting", "*", "*")
        }
        
        # Get all children from waiting and served
        all_children = {child for child, _ in waiting_children} | served_children
        unserved_children = all_children - served_children
        
        # Get current sandwiches and their status
        sandwiches_on_tray = {
            get_parts(fact)[1] 
            for fact in state 
            if match(fact, "ontray", "*", "*")
        }
        sandwiches_at_kitchen = {
            get_parts(fact)[1] 
            for fact in state 
            if match(fact, "at_kitchen_sandwich", "*")
        }
        gluten_free_sandwiches = {
            get_parts(fact)[1] 
            for fact in state 
            if match(fact, "no_gluten_sandwich", "*")
        }
        all_sandwiches = sandwiches_on_tray | sandwiches_at_kitchen
        
        # Get tray locations
        tray_locations = {
            get_parts(fact)[1]: get_parts(fact)[2] 
            for fact in state 
            if match(fact, "at", "tray*", "*")
        }
        
        total_cost = 0
        
        # For each unserved child
        for child in unserved_children:
            # Find the child's location
            child_loc = next((loc for (c, loc) in waiting_children if c == child), None)
            
            allergic = child in self.allergic_children
            suitable_sandwich = None
            
            # Check for existing suitable sandwiches
            if allergic:
                # Need gluten-free sandwich
                suitable_sandwiches = gluten_free_sandwiches & all_sandwiches
            else:
                # Can use any sandwich
                suitable_sandwiches = all_sandwiches
            
            # Check if any suitable sandwich is on a tray at the child's location
            for s in suitable_sandwiches:
                if s in sandwiches_on_tray:
                    # Find which tray it's on
                    tray = next(
                        get_parts(fact)[2] 
                        for fact in state 
                        if match(fact, "ontray", s, "*")
                    )
                    # Check if tray is at child's location
                    if tray_locations.get(tray) == child_loc:
                        suitable_sandwich = s
                        break
            
            if suitable_sandwich:
                # Just need to serve (1 action)
                total_cost += 1
            else:
                # Need to make sandwich or put existing one on tray
                if suitable_sandwiches:
                    # Use existing sandwich
                    sandwich = next(iter(suitable_sandwiches))
                    if sandwich in sandwiches_at_kitchen:
                        # Need to put on tray (1 action)
                        total_cost += 1
                    # Then serve (1 action)
                    total_cost += 1
                else:
                    # Need to make new sandwich
                    total_cost += 1  # make_sandwich action
                    total_cost += 1  # put_on_tray action
                    total_cost += 1  # serve action
        
        # Estimate tray movements (1 per unique location with unserved children)
        unserved_locations = {
            loc for (child, loc) in waiting_children 
            if child in unserved_children
        }
        total_cost += len(unserved_locations)  # move_tray actions
        
        return total_cost
