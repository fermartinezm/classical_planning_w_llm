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
    - Whether sandwiches need to be placed on trays
    - Whether trays need to be moved
    - Special handling for gluten-free sandwiches

    # Assumptions:
    - Making a sandwich takes 1 action
    - Putting a sandwich on a tray takes 1 action
    - Serving a sandwich takes 1 action
    - Moving a tray takes 1 action
    - Gluten-free sandwiches require special handling

    # Heuristic Initialization
    - Extract information about which children are allergic to gluten
    - Extract waiting locations of children
    - Extract goal conditions (which children need to be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. Count how many children still need to be served
    2. For each unserved child:
       a. If allergic to gluten:
          - Check if there's a gluten-free sandwich on a tray at their location
          - If not, check if one can be made from available ingredients
          - If ingredients are available, count making and placing on tray
          - If no ingredients, this is an unsolvable state (return infinity)
       b. If not allergic:
          - Check if there's any sandwich on a tray at their location
          - If not, check if one can be made from available ingredients
          - If ingredients are available, count making and placing on tray
          - If no ingredients, this is an unsolvable state (return infinity)
    3. Count tray movements needed (if tray isn't already at the right location)
    4. Sum all actions needed
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
        
        # Extract waiting locations of children
        self.child_locations = {}
        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                self.child_locations[parts[1]] = parts[2]
        
        # Extract goal children (who need to be served)
        self.goal_children = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                self.goal_children.add(get_parts(goal)[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if goal is already reached
        if self.goals <= state:
            return 0
            
        # Initialize counters
        total_cost = 0
        unserved_children = set()
        
        # Find unserved children
        for child in self.goal_children:
            if f"(served {child})" not in state:
                unserved_children.add(child)
        
        # Check for each unserved child what needs to be done
        for child in unserved_children:
            allergic = child in self.allergic_children
            location = self.child_locations[child]
            
            # Check if there's already a suitable sandwich on a tray at the location
            sandwich_found = False
            for fact in state:
                if match(fact, "ontray", "*", "*"):
                    sandwich, tray = get_parts(fact)[1:]
                    # Check tray location
                    tray_loc_fact = f"(at {tray} {location})"
                    if tray_loc_fact in state:
                        # Check if sandwich is suitable
                        if not allergic:
                            sandwich_found = True
                            break
                        elif f"(no_gluten_sandwich {sandwich})" in state:
                            sandwich_found = True
                            break
            
            if sandwich_found:
                # Just need to serve (1 action)
                total_cost += 1
                continue
                
            # No suitable sandwich found - need to make one
            # First check if we have ingredients
            if allergic:
                # Need gluten-free bread and content
                bread_available = any(
                    f"(at_kitchen_bread {bread})" in state and 
                    f"(no_gluten_bread {bread})" in self.static
                    for bread in [get_parts(fact)[1] for fact in state 
                                if match(fact, "at_kitchen_bread", "*")]
                )
                content_available = any(
                    f"(at_kitchen_content {content})" in state and 
                    f"(no_gluten_content {content})" in self.static
                    for content in [get_parts(fact)[1] for fact in state 
                                  if match(fact, "at_kitchen_content", "*")]
                )
            else:
                # Any bread and content will do
                bread_available = any(
                    f"(at_kitchen_bread {bread})" in state
                    for bread in [get_parts(fact)[1] for fact in state 
                                if match(fact, "at_kitchen_bread", "*")]
                )
                content_available = any(
                    f"(at_kitchen_content {content})" in state
                    for content in [get_parts(fact)[1] for fact in state 
                                  if match(fact, "at_kitchen_content", "*")]
                )
            
            if not bread_available or not content_available:
                # No ingredients available - unsolvable state
                return float('inf')
            
            # We have ingredients - count making sandwich (1 action)
            total_cost += 1
            
            # Need to put sandwich on tray (1 action)
            total_cost += 1
            
            # Check if tray is at kitchen (where sandwiches are made)
            tray_at_kitchen = any(
                f"(at {tray} kitchen)" in state
                for tray in [get_parts(fact)[1] for fact in state 
                          if match(fact, "tray", "*")]
            )
            
            if not tray_at_kitchen:
                # Need to move tray to kitchen first (1 action)
                total_cost += 1
            
            # Need to move tray to child's location (1 action)
            total_cost += 1
            
            # Finally serve the sandwich (1 action)
            total_cost += 1
        
        return total_cost
