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
    - Sandwiches that need to be made (from available bread and content)
    - Sandwiches that need to be placed on trays
    - Trays that need to be moved to children's locations
    - Special requirements for gluten-allergic children

    # Assumptions:
    - All children must be served (goal condition)
    - Gluten-free sandwiches must be made for allergic children
    - Regular sandwiches can be made for non-allergic children
    - Each sandwich requires one bread and one content portion
    - Trays can hold multiple sandwiches
    - Kitchen is the starting location for all trays

    # Heuristic Initialization
    - Extract information about allergic children from static facts
    - Identify waiting children and their locations from static facts
    - Track gluten-free requirements for bread and content portions

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (those not yet in goal state)
    2. For each unserved child:
       - If allergic: check if suitable gluten-free sandwich exists or can be made
       - If not allergic: check if any sandwich exists or can be made
    3. Estimate actions needed:
       - Making sandwiches (1 action per sandwich)
       - Putting sandwiches on trays (1 action per sandwich)
       - Moving trays to children's locations (1 action per tray move)
       - Serving sandwiches (1 action per child)
    4. Optimize by:
       - Grouping sandwiches for same location on same tray
       - Prioritizing gluten-free sandwiches for allergic children
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
        
        # Extract waiting children and their locations
        self.waiting_children = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in self.static 
            if match(fact, "waiting", "*", "*")
        }
        
        # Extract gluten-free bread and content
        self.no_gluten_breads = {
            get_parts(fact)[1] 
            for fact in self.static 
            if match(fact, "no_gluten_bread", "*")
        }
        
        self.no_gluten_contents = {
            get_parts(fact)[1] 
            for fact in self.static 
            if match(fact, "no_gluten_content", "*")
        }

    def __call__(self, node):
        """Estimate the minimum cost to serve all children."""
        state = node.state
        
        # Count unserved children (those not in goal state)
        unserved_children = {
            child for child in self.waiting_children
            if f"(served {child})" not in state
        }
        
        if not unserved_children:
            return 0  # All children served
        
        # Count available resources
        available_breads = {
            get_parts(fact)[1] 
            for fact in state 
            if match(fact, "at_kitchen_bread", "*")
        }
        
        available_contents = {
            get_parts(fact)[1] 
            for fact in state 
            if match(fact, "at_kitchen_content", "*")
        }
        
        available_sandwiches = {
            get_parts(fact)[1] 
            for fact in state 
            if match(fact, "at_kitchen_sandwich", "*")
        }
        
        gluten_free_sandwiches = {
            get_parts(fact)[1] 
            for fact in state 
            if match(fact, "no_gluten_sandwich", "*")
        }
        
        # Count sandwiches on trays
        sandwiches_on_trays = {
            get_parts(fact)[1] 
            for fact in state 
            if match(fact, "ontray", "*", "*")
        }
        
        # Count trays not in kitchen
        trays_not_in_kitchen = {
            get_parts(fact)[1] 
            for fact in state 
            if match(fact, "at", "*", "*") and not match(fact, "at", "*", "kitchen")
        }
        
        # Initialize cost
        total_cost = 0
        
        # For each unserved child, estimate actions needed
        for child in unserved_children:
            is_allergic = child in self.allergic_children
            child_location = self.waiting_children[child]
            
            if is_allergic:
                # Need gluten-free sandwich
                if not gluten_free_sandwiches:
                    # Need to make one
                    if (self.no_gluten_breads & available_breads and 
                        self.no_gluten_contents & available_contents):
                        total_cost += 1  # make_sandwich_no_gluten
                        # Assume we'll use one bread and one content
                        available_breads -= set(next(iter(self.no_gluten_breads & available_breads)))
                        available_contents -= set(next(iter(self.no_gluten_contents & available_contents)))
                    else:
                        # No ingredients available - problem can't be solved
                        return float('inf')
            else:
                # Can use any sandwich
                if not (available_sandwiches or sandwiches_on_trays):
                    # Need to make one
                    if available_breads and available_contents:
                        total_cost += 1  # make_sandwich
                        # Assume we'll use one bread and one content
                        available_breads.pop()
                        available_contents.pop()
                    else:
                        # No ingredients available - problem can't be solved
                        return float('inf')
            
            # Check if tray needs to be moved to child's location
            tray_at_location = any(
                match(fact, "at", "*", child_location)
                for fact in state
            )
            
            if not tray_at_location:
                total_cost += 1  # move_tray
            
            # Serving the sandwich
            total_cost += 1  # serve_sandwich or serve_sandwich_no_gluten
        
        return total_cost
