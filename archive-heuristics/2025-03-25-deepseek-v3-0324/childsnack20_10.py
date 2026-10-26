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
    - The number of sandwiches that need to be made
    - The number of sandwiches that need to be placed on trays
    - The number of trays that need to be moved
    - The number of sandwiches that need to be served

    # Assumptions:
    - Each child needs exactly one sandwich
    - Gluten-free sandwiches must be served to allergic children
    - Regular sandwiches can be served to any child
    - Multiple sandwiches can be placed on a single tray
    - Trays start in the kitchen and may need to be moved to serve children

    # Heuristic Initialization
    - Extract information about allergic children from static facts
    - Identify goal conditions (which children need to be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (those not yet in goal state)
    2. For each unserved child:
       a) If allergic, check if a gluten-free sandwich is available on a tray at their location
       b) If not allergic, check if any sandwich is available on a tray at their location
    3. For missing sandwiches:
       a) Count bread-content pairs available in kitchen
       b) Add actions needed to make sandwiches (1 per sandwich)
    4. For sandwiches not on trays:
       a) Add actions needed to put them on trays (1 per sandwich)
    5. For trays not at correct locations:
       a) Add actions needed to move them (1 per required move)
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
        
        # Extract waiting children and their locations from static facts
        self.waiting_children = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in self.static 
            if match(fact, "waiting", "*", "*")
        }

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic_value = 0
        
        # 1. Identify unserved children (those not in goal state)
        served_children = {
            get_parts(fact)[1] for fact in state 
            if match(fact, "served", "*")
        }
        unserved_children = {
            child for child in self.waiting_children 
            if f"(served {child})" not in self.goals or f"(served {child})" not in state
        }
        
        # 2. Check sandwich requirements for each unserved child
        needed_gluten_free = 0
        needed_regular = 0
        
        for child in unserved_children:
            if child in self.allergic_children:
                needed_gluten_free += 1
            else:
                needed_regular += 1
        
        # 3. Count available resources in current state
        available_bread = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*"))
        available_content = sum(1 for fact in state if match(fact, "at_kitchen_content", "*"))
        available_sandwiches_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*"))
        available_sandwiches_trays = sum(1 for fact in state if match(fact, "ontray", "*", "*"))
        available_gluten_free_sandwiches = sum(1 for fact in state if match(fact, "no_gluten_sandwich", "*"))
        
        # 4. Calculate sandwich making actions needed
        # Gluten-free sandwiches
        missing_gluten_free = max(0, needed_gluten_free - available_gluten_free_sandwiches)
        possible_gluten_free = min(
            sum(1 for fact in state if match(fact, "no_gluten_bread", "*")),
            sum(1 for fact in state if match(fact, "no_gluten_content", "*")),
            missing_gluten_free
        )
        heuristic_value += possible_gluten_free  # make_sandwich_no_gluten actions
        
        # Regular sandwiches
        total_needed = needed_regular + max(0, missing_gluten_free - possible_gluten_free)
        available_pairs = min(
            available_bread - possible_gluten_free,
            available_content - possible_gluten_free,
            total_needed - available_sandwiches_kitchen - available_sandwiches_trays
        )
        heuristic_value += max(0, available_pairs)  # make_sandwich actions
        
        # 5. Calculate tray operations
        # Sandwiches not on trays
        sandwiches_to_tray = max(0, available_sandwiches_kitchen)
        heuristic_value += sandwiches_to_tray  # put_on_tray actions
        
        # 6. Calculate tray movements
        # For each tray, check if it needs to be moved to serve children
        tray_locations = {}
        for fact in state:
            if match(fact, "at", "tray*", "*"):
                _, tray, loc = get_parts(fact)
                tray_locations[tray] = loc
        
        # For each unserved child, check if a tray with appropriate sandwich is at their location
        for child in unserved_children:
            child_loc = self.waiting_children[child]
            tray_at_location = any(
                loc == child_loc for tray, loc in tray_locations.items()
            )
            if not tray_at_location:
                heuristic_value += 1  # move_tray action
        
        # 7. Calculate serving actions (1 per child)
        heuristic_value += len(unserved_children)  # serve_sandwich actions
        
        return heuristic_value
