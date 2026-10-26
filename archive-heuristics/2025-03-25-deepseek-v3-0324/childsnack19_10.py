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
    2. Estimating the work needed to create sandwiches for them
    3. Accounting for tray movements and sandwich placements

    # Assumptions:
    - Each child needs exactly one sandwich
    - Gluten-free sandwiches require specific bread and content
    - Trays can hold multiple sandwiches
    - All bread and content starts in the kitchen
    - Kitchen is the only place where sandwiches can be made

    # Heuristic Initialization
    - Extract information about children's allergies from static facts
    - Identify gluten-free bread and content requirements
    - Note which children are waiting where

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (goal conditions not met)
    2. For each unserved child:
       a) If allergic to gluten:
          - Check if suitable gluten-free sandwich exists or can be made
          - Count required make_sandwich_no_gluten action
       b) If not allergic:
          - Check if any sandwich exists or can be made
          - Count required make_sandwich action
    3. For each sandwich needed:
       - Count put_on_tray action if not already on tray
    4. For each tray movement needed:
       - Count move_tray action if tray isn't at child's location
    5. For each sandwich serving:
       - Count serve_sandwich(_no_gluten) action
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract allergy information from static facts
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

    def __call__(self, node):
        """Estimate the number of actions needed to serve all children."""
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
        sandwiches_on_trays = set()
        tray_locations = {}  # tray -> location
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at_kitchen_bread", "*"):
                available_breads.add(parts[1])
            elif match(fact, "at_kitchen_content", "*"):
                available_contents.add(parts[1])
            elif match(fact, "at_kitchen_sandwich", "*"):
                available_sandwiches.add(parts[1])
            elif match(fact, "no_gluten_sandwich", "*"):
                gluten_free_sandwiches.add(parts[1])
            elif match(fact, "ontray", "*", "*"):
                sandwiches_on_trays.add(parts[1])
            elif match(fact, "at", "*", "*"):
                tray_locations[parts[1]] = parts[2]
        
        # Count children waiting at each location
        waiting_at = {}  # location -> set of children
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                child, location = get_parts(fact)[1], get_parts(fact)[2]
                if child in unserved_children:
                    waiting_at.setdefault(location, set()).add(child)
        
        total_cost = 0
        
        # Process each location where children are waiting
        for location, children in waiting_at.items():
            # Count sandwiches needed at this location
            sandwiches_needed = len(children)
            
            # Count sandwiches already on trays at this location
            available_at_location = 0
            for tray, tray_loc in tray_locations.items():
                if tray_loc == location:
                    # Count sandwiches on this tray that can serve children here
                    for fact in state:
                        if match(fact, "ontray", "*", tray):
                            sandwich = get_parts(fact)[1]
                            # Check if sandwich can serve any child here
                            for child in children:
                                if child in self.allergic_children:
                                    if sandwich in gluten_free_sandwiches:
                                        available_at_location += 1
                                        break
                                else:
                                    available_at_location += 1
                                    break
            
            sandwiches_needed -= available_at_location
            if sandwiches_needed <= 0:
                continue
            
            # Estimate cost to make and deliver needed sandwiches
            sandwiches_to_make = sandwiches_needed
            gluten_free_needed = sum(1 for c in children if c in self.allergic_children)
            
            # Cost to make sandwiches
            while sandwiches_to_make > 0 and gluten_free_needed > 0:
                if self.gluten_free_breads & available_breads and self.gluten_free_contents & available_contents:
                    total_cost += 1  # make_sandwich_no_gluten
                    gluten_free_needed -= 1
                    sandwiches_to_make -= 1
                    # Assume we use one bread and one content
                    available_breads -= set(self.gluten_free_breads)  # remove one
                    available_contents -= set(self.gluten_free_contents)  # remove one
            
            while sandwiches_to_make > 0:
                if available_breads and available_contents:
                    total_cost += 1  # make_sandwich
                    sandwiches_to_make -= 1
                    # Assume we use one bread and one content
                    available_breads.pop()  # remove one
                    available_contents.pop()  # remove one
            
            # Cost to put sandwiches on trays (at least one action per sandwich)
            total_cost += sandwiches_needed  # put_on_tray
            
            # Cost to move trays if not already at location
            if location != "kitchen":
                total_cost += 1  # move_tray
            
            # Cost to serve sandwiches (one per child)
            total_cost += len(children)  # serve_sandwich(_no_gluten)
        
        return total_cost
