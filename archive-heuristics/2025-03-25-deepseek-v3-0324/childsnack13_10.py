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
    2. Estimating sandwich preparation needs (including gluten-free requirements)
    3. Considering tray movements and sandwich placements

    # Assumptions:
    - Each child needs exactly one sandwich
    - Gluten-allergic children must be served gluten-free sandwiches
    - Sandwiches can be made in parallel if ingredients are available
    - Tray movements are needed when serving children not in the kitchen

    # Heuristic Initialization
    - Extract allergic children from static facts
    - Extract waiting locations from static facts
    - Identify gluten-free bread and content requirements

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (those without '(served childX)' in state)
    2. For each unserved child:
       a) If allergic, check if suitable gluten-free sandwich exists or can be made
       b) If not allergic, check if any sandwich exists or can be made
    3. Estimate sandwich preparation actions:
       - 1 action per sandwich to make (either regular or gluten-free)
       - 1 action to put sandwich on tray
    4. Estimate tray movement actions:
       - 1 action per tray movement between kitchen and child's location
    5. Estimate serving actions:
       - 1 action per child to serve their sandwich
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
        
        # Extract waiting locations
        self.child_locations = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in self.static
            if match(fact, "waiting", "*", "*")
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
        
        # Check if goal is already reached
        if self.goals <= state:
            return 0
            
        # Count unserved children
        served_children = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "served", "*")
        }
        
        unserved_children = set()
        for child in self.child_locations:
            if child not in served_children:
                unserved_children.add(child)
        
        total_cost = 0
        
        # Track sandwich resources
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
        
        sandwiches_on_trays = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "ontray", "*", "*")
        }
        
        gluten_free_sandwiches = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "no_gluten_sandwich", "*")
        }
        
        # Count trays in kitchen
        trays_in_kitchen = sum(
            1 for fact in state 
            if match(fact, "at", "*", "kitchen") and not match(fact, "at_kitchen_*", "*")
        )
        
        # Process each unserved child
        for child in unserved_children:
            is_allergic = child in self.allergic_children
            location = self.child_locations[child]
            
            # Check if suitable sandwich exists on tray at location
            sandwich_found = False
            for fact in state:
                if match(fact, "ontray", "*", "*") and match(fact, "*", "*", location):
                    sandwich = get_parts(fact)[1]
                    if not is_allergic or sandwich in gluten_free_sandwiches:
                        sandwich_found = True
                        break
            
            if sandwich_found:
                total_cost += 1  # serve action
                continue
                
            # Need to make and deliver a sandwich
            if is_allergic:
                # Need gluten-free sandwich
                if available_breads & self.gluten_free_breads and available_contents & self.gluten_free_contents:
                    total_cost += 1  # make_sandwich_no_gluten
                else:
                    # No suitable ingredients - high penalty
                    total_cost += 100
                    continue
            else:
                # Can use any sandwich
                if available_breads and available_contents:
                    total_cost += 1  # make_sandwich
                else:
                    # No ingredients - high penalty
                    total_cost += 100
                    continue
            
            # Put on tray
            total_cost += 1
            
            # Move tray if needed
            if location != "kitchen":
                total_cost += 1
                
            # Serve sandwich
            total_cost += 1
            
            # Update resource counts (approximate)
            if available_breads:
                available_breads.pop()
            if available_contents:
                available_contents.pop()
            if trays_in_kitchen > 0:
                trays_in_kitchen -= 1
        
        return total_cost
