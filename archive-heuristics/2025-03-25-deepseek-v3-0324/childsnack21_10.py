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
    2. Estimating the sandwich-making and serving actions required for each
    3. Accounting for gluten-free requirements where needed
    4. Considering tray movements when necessary

    # Assumptions:
    - All children must be served (goal condition)
    - Gluten-free sandwiches require specific bread and content portions
    - Sandwiches must be made, placed on trays, and trays moved before serving
    - Multiple sandwiches can be served from the same tray if at the correct location

    # Heuristic Initialization
    - Extract static information about children's allergies and waiting locations
    - Identify goal conditions (all children served)
    - Precompute mappings for gluten-free requirements

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (main driver of heuristic value)
    2. For each unserved child:
       a) If allergic to gluten:
          - Check if suitable gluten-free sandwich exists or can be made
          - Count required make_sandwich_no_gluten and put_on_tray actions
       b) If not allergic:
          - Check if any sandwich exists or can be made
          - Count required make_sandwich and put_on_tray actions
    3. Estimate tray movements:
       - If tray needs to move to serve a child, count move_tray action
    4. Count serve actions (one per child)
    5. Sum all estimated actions
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
        
        # Extract waiting locations for children
        self.child_waiting_locations = {}
        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                self.child_waiting_locations[parts[1]] = parts[2]

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Count unserved children
        unserved_children = set()
        for child in self.allergic_children | self.normal_children:
            if f"(served {child})" not in state:
                unserved_children.add(child)
        
        if not unserved_children:
            return 0  # Goal state
        
        # Initialize action counters
        make_actions = 0
        tray_actions = 0
        move_actions = 0
        serve_actions = len(unserved_children)
        
        # Track available sandwiches
        available_sandwiches = set()
        gluten_free_sandwiches = set()
        sandwiches_on_trays = set()
        
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                available_sandwiches.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_sandwich", "*"):
                gluten_free_sandwiches.add(get_parts(fact)[1])
            elif match(fact, "ontray", "*", "*"):
                sandwiches_on_trays.add(get_parts(fact)[1])
        
        # Check tray locations
        tray_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1].startswith("tray"):
                    tray_locations[parts[1]] = parts[2]
        
        # For each unserved child, estimate required actions
        for child in unserved_children:
            waiting_location = self.child_waiting_locations[child]
            needs_gluten_free = child in self.allergic_children
            
            # Check if suitable sandwich is already on a tray at the right location
            suitable_sandwich_found = False
            for tray, location in tray_locations.items():
                if location == waiting_location:
                    for sandwich in sandwiches_on_trays:
                        if (not needs_gluten_free) or (sandwich in gluten_free_sandwiches):
                            suitable_sandwich_found = True
                            break
                    if suitable_sandwich_found:
                        break
            
            if not suitable_sandwich_found:
                # Need to make a new sandwich if none available
                if needs_gluten_free:
                    if not gluten_free_sandwiches - sandwiches_on_trays:
                        make_actions += 1  # Need to make a gluten-free sandwich
                else:
                    if not available_sandwiches - sandwiches_on_trays:
                        make_actions += 1  # Need to make a regular sandwich
                
                # Need to put sandwich on tray
                tray_actions += 1
                
                # Check if tray needs to be moved
                if not any(loc == waiting_location for loc in tray_locations.values()):
                    move_actions += 1
        
        # Total heuristic is sum of all estimated actions
        return make_actions + tray_actions + move_actions + serve_actions
