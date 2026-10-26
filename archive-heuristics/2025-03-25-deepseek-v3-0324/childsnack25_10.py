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
    - Sandwiches that need to be made (including gluten-free ones)
    - Sandwiches that need to be placed on trays
    - Trays that need to be moved to children's locations
    - Sandwiches that need to be served to children

    # Assumptions:
    - Each child needs exactly one sandwich
    - Gluten-allergic children must be served gluten-free sandwiches
    - Regular children can be served any sandwich
    - Multiple sandwiches can be placed on a single tray
    - The kitchen is the only starting location for trays

    # Heuristic Initialization
    - Extract information about which children are allergic to gluten
    - Extract waiting locations of children
    - Identify gluten-free bread and content portions

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (both allergic and non-allergic)
    2. For each unserved allergic child:
       - If no suitable gluten-free sandwich exists on a tray at their location:
         - If no gluten-free sandwich exists in kitchen, count actions to make one
         - Count action to put it on a tray
         - Count action to move tray to child's location
       - Count action to serve the sandwich
    3. For each unserved non-allergic child:
       - If no sandwich exists on a tray at their location:
         - If no sandwich exists in kitchen, count actions to make one
         - Count action to put it on a tray
         - Count action to move tray to child's location
       - Count action to serve the sandwich
    4. Optimize by considering that:
       - Multiple sandwiches can be made in parallel
       - Multiple sandwiches can be placed on same tray
       - Multiple children at same location can be served from same tray
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
        
        # Extract waiting locations of children
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
        """Estimate the number of actions needed to serve all children."""
        state = node.state
        
        # Count served and unserved children
        served_children = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "served", "*")
        }
        unserved_children = set(self.child_locations.keys()) - served_children
        
        # Initialize action count
        total_actions = 0
        
        # Track sandwiches in kitchen, on trays, and gluten-free status
        sandwiches_in_kitchen = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "at_kitchen_sandwich", "*")
        }
        gluten_free_sandwiches = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "no_gluten_sandwich", "*")
        }
        
        # Track sandwiches on trays and their locations
        sandwiches_on_trays = {}
        tray_locations = {}
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1:]
                sandwiches_on_trays[s] = t
            elif match(fact, "at", "*", "*"):
                t, p = get_parts(fact)[1:]
                if t.startswith("tray"):
                    tray_locations[t] = p
        
        # Process allergic children first (more constrained)
        for child in unserved_children:
            if child not in self.allergic_children:
                continue
                
            location = self.child_locations[child]
            
            # Check if there's a gluten-free sandwich on a tray at child's location
            suitable_sandwich = None
            for s in gluten_free_sandwiches:
                tray = sandwiches_on_trays.get(s)
                if tray and tray_locations.get(tray) == location:
                    suitable_sandwich = s
                    break
            
            if suitable_sandwich:
                total_actions += 1  # serve_sandwich_no_gluten
                continue
                
            # Need to make and deliver a gluten-free sandwich
            if not (sandwiches_in_kitchen & gluten_free_sandwiches):
                total_actions += 1  # make_sandwich_no_gluten
                
            # Find a tray at kitchen (or assume we can use one)
            tray_at_kitchen = None
            for t, loc in tray_locations.items():
                if loc == "kitchen":
                    tray_at_kitchen = t
                    break
            
            total_actions += 1  # put_on_tray
            if location != "kitchen":
                total_actions += 1  # move_tray
            total_actions += 1  # serve_sandwich_no_gluten
        
        # Process non-allergic children
        for child in unserved_children:
            if child in self.allergic_children:
                continue
                
            location = self.child_locations[child]
            
            # Check if there's any sandwich on a tray at child's location
            suitable_sandwich = None
            for s in sandwiches_on_trays:
                tray = sandwiches_on_trays[s]
                if tray and tray_locations.get(tray) == location:
                    suitable_sandwich = s
                    break
            
            if suitable_sandwich:
                total_actions += 1  # serve_sandwich
                continue
                
            # Need to make and deliver a sandwich
            if not sandwiches_in_kitchen:
                total_actions += 1  # make_sandwich
                
            # Find a tray at kitchen (or assume we can use one)
            tray_at_kitchen = None
            for t, loc in tray_locations.items():
                if loc == "kitchen":
                    tray_at_kitchen = t
                    break
            
            total_actions += 1  # put_on_tray
            if location != "kitchen":
                total_actions += 1  # move_tray
            total_actions += 1  # serve_sandwich
        
        return total_actions
