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
    - Special handling for gluten-allergic children

    # Assumptions:
    - All children must be served (goal condition)
    - Gluten-free sandwiches must be made for allergic children
    - Regular sandwiches can be made for non-allergic children
    - Each sandwich can only serve one child
    - Trays can carry multiple sandwiches

    # Heuristic Initialization
    - Extract information about allergic children from static facts
    - Identify waiting children and their locations
    - Track available bread and content portions

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (from goal conditions)
    2. For each unserved child:
       - If allergic: check if suitable gluten-free sandwich exists or can be made
       - If not allergic: check if any sandwich exists or can be made
    3. Estimate actions needed:
       - Making sandwiches (1 action per sandwich)
       - Putting sandwiches on trays (1 action per sandwich)
       - Moving trays to children's locations (1 action per tray move)
       - Serving sandwiches (1 action per child)
    4. Special cases:
       - If sandwich is already on tray at correct location: just serve
       - If tray needs to be moved: account for move action
       - If sandwich needs to be made: account for make and put actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract allergic children from static facts
        self.allergic_children = {
            get_parts(fact)[1] for fact in self.static 
            if match(fact, "allergic_gluten", "*")
        }
        
        # Extract non-allergic children from static facts
        self.non_allergic_children = {
            get_parts(fact)[1] for fact in self.static 
            if match(fact, "not_allergic_gluten", "*")
        }

    def __call__(self, node):
        """Estimate the number of actions needed to serve all children."""
        state = node.state
        
        # Count unserved children (goal is to serve all)
        unserved_children = {
            get_parts(goal)[1] for goal in self.goals 
            if match(goal, "served", "*")
        }
        
        # Get waiting children and their locations
        waiting_children = {}
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                _, child, place = get_parts(fact)
                waiting_children[child] = place
        
        # Track available resources
        kitchen_breads = {
            get_parts(fact)[1] for fact in state 
            if match(fact, "at_kitchen_bread", "*")
        }
        kitchen_contents = {
            get_parts(fact)[1] for fact in state 
            if match(fact, "at_kitchen_content", "*")
        }
        kitchen_sandwiches = {
            get_parts(fact)[1] for fact in state 
            if match(fact, "at_kitchen_sandwich", "*")
        }
        gluten_free_breads = {
            get_parts(fact)[1] for fact in state 
            if match(fact, "no_gluten_bread", "*")
        }
        gluten_free_contents = {
            get_parts(fact)[1] for fact in state 
            if match(fact, "no_gluten_content", "*")
        }
        gluten_free_sandwiches = {
            get_parts(fact)[1] for fact in state 
            if match(fact, "no_gluten_sandwich", "*")
        }
        
        # Get sandwiches on trays and their tray locations
        sandwiches_on_trays = {}
        tray_locations = {}
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                _, sandwich, tray = get_parts(fact)
                sandwiches_on_trays[sandwich] = tray
            elif match(fact, "at", "*", "*"):
                _, tray, place = get_parts(fact)
                tray_locations[tray] = place
        
        total_cost = 0
        
        for child in unserved_children:
            if child not in waiting_children:
                continue  # Child not waiting to be served
                
            child_location = waiting_children[child]
            served = False
            
            # Check if there's already a suitable sandwich on a tray at the location
            for sandwich, tray in sandwiches_on_trays.items():
                if tray in tray_locations and tray_locations[tray] == child_location:
                    if child in self.allergic_children:
                        if sandwich in gluten_free_sandwiches:
                            total_cost += 1  # serve_sandwich_no_gluten action
                            served = True
                            break
                    else:
                        total_cost += 1  # serve_sandwich action
                        served = True
                        break
            
            if served:
                continue
                
            # If no suitable sandwich on tray at location, we need to:
            # 1. Make sandwich (if not already made)
            # 2. Put on tray
            # 3. Move tray (if not at location)
            # 4. Serve
            
            if child in self.allergic_children:
                # Need gluten-free sandwich
                if not gluten_free_sandwiches and not gluten_free_breads:
                    return float('inf')  # No way to make required sandwich
                
                total_cost += 1  # make_sandwich_no_gluten or use existing
                
                # Find a tray that can be used
                available_trays = [
                    tray for tray, loc in tray_locations.items() 
                    if loc == "kitchen"
                ]
                if not available_trays:
                    return float('inf')  # No available trays
                
                total_cost += 1  # put_on_tray
                
                # Check if tray needs to be moved
                tray = available_trays[0]
                if tray_locations[tray] != child_location:
                    total_cost += 1  # move_tray
                
                total_cost += 1  # serve_sandwich_no_gluten
            else:
                # Can use any sandwich
                if not kitchen_sandwiches and not kitchen_breads:
                    return float('inf')  # No way to make sandwich
                
                total_cost += 1  # make_sandwich or use existing
                
                # Find a tray that can be used
                available_trays = [
                    tray for tray, loc in tray_locations.items() 
                    if loc == "kitchen"
                ]
                if not available_trays:
                    return float('inf')  # No available trays
                
                total_cost += 1  # put_on_tray
                
                # Check if tray needs to be moved
                tray = available_trays[0]
                if tray_locations[tray] != child_location:
                    total_cost += 1  # move_tray
                
                total_cost += 1  # serve_sandwich
        
        return total_cost
