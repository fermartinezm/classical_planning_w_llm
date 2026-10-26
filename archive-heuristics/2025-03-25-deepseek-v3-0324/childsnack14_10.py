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
    2. Accounting for sandwiches that need to be made (considering gluten allergies)
    3. Considering tray movements and sandwich placements

    # Assumptions:
    - Each child needs exactly one sandwich
    - Gluten-allergic children must be served gluten-free sandwiches
    - Sandwiches can be made in any order as long as requirements are met
    - Tray movements between locations are necessary when serving

    # Heuristic Initialization
    - Extract information about children's allergies and waiting locations
    - Identify gluten-free bread and content portions
    - Store goal conditions (which children need to be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (main component of heuristic)
    2. For each unserved child:
       - If allergic to gluten:
         - Check if a gluten-free sandwich exists on a tray at their location
         - If not, check if one can be made from available gluten-free ingredients
         - Add costs for making and placing sandwich if needed
       - If not allergic:
         - Check if any sandwich exists on a tray at their location
         - If not, check if one can be made from available ingredients
         - Add costs for making and placing sandwich if needed
    3. Add costs for moving trays if needed
    4. The total heuristic is the sum of:
       - 1 for each unserved child (serving action)
       - 1 for each sandwich that needs to be made
       - 1 for each sandwich that needs to be placed on tray
       - 1 for each tray movement needed
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

        # Extract gluten-free ingredients
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()
        for fact in self.static:
            if match(fact, "no_gluten_bread", "*"):
                self.gluten_free_breads.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_content", "*"):
                self.gluten_free_contents.add(get_parts(fact)[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        served_children = set()
        waiting_children = set()
        allergic_waiting = set()
        normal_waiting = set()
        
        # Current sandwiches on trays at locations
        sandwiches_on_trays = {}
        # Sandwiches in kitchen ready to be placed
        sandwiches_in_kitchen = set()
        # Gluten-free sandwiches
        gluten_free_sandwiches = set()
        # Tray locations
        tray_locations = {}
        # Available bread and content in kitchen
        available_bread = set()
        available_content = set()
        gluten_free_bread_available = set()
        gluten_free_content_available = set()

        # Analyze current state
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "served", "*"):
                served_children.add(parts[1])
            elif match(fact, "waiting", "*", "*"):
                child = parts[1]
                waiting_children.add(child)
                if child in self.allergic_children:
                    allergic_waiting.add(child)
                else:
                    normal_waiting.add(child)
            elif match(fact, "ontray", "*", "*"):
                sandwich, tray = parts[1], parts[2]
                sandwiches_on_trays[(sandwich, tray)] = True
            elif match(fact, "at_kitchen_sandwich", "*"):
                sandwiches_in_kitchen.add(parts[1])
            elif match(fact, "no_gluten_sandwich", "*"):
                gluten_free_sandwiches.add(parts[1])
            elif match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith("tray"):
                    tray_locations[obj] = loc
            elif match(fact, "at_kitchen_bread", "*"):
                bread = parts[1]
                available_bread.add(bread)
                if bread in self.gluten_free_breads:
                    gluten_free_bread_available.add(bread)
            elif match(fact, "at_kitchen_content", "*"):
                content = parts[1]
                available_content.add(content)
                if content in self.gluten_free_contents:
                    gluten_free_content_available.add(content)

        # Calculate heuristic components
        unserved = waiting_children - served_children
        h = 0

        for child in unserved:
            # Each child needs at least one action to be served
            h += 1
            
            is_allergic = child in self.allergic_children
            # Find the child's location
            child_loc = None
            for fact in state:
                if match(fact, "waiting", child, "*"):
                    child_loc = get_parts(fact)[2]
                    break
            
            # Check if there's a suitable sandwich on a tray at child's location
            suitable_sandwich_found = False
            for (sandwich, tray), _ in sandwiches_on_trays.items():
                tray_loc = tray_locations.get(tray, None)
                if tray_loc == child_loc:
                    if not is_allergic or sandwich in gluten_free_sandwiches:
                        suitable_sandwich_found = True
                        break
            
            if not suitable_sandwich_found:
                # Need to make and place a sandwich
                if is_allergic:
                    # Need gluten-free sandwich
                    if gluten_free_bread_available and gluten_free_content_available:
                        h += 1  # make_sandwich_no_gluten action
                    else:
                        # No ingredients available - high penalty
                        h += 3
                else:
                    # Any sandwich will do
                    if available_bread and available_content:
                        h += 1  # make_sandwich action
                    else:
                        # No ingredients available - high penalty
                        h += 3
                
                # Need to place sandwich on tray
                h += 1  # put_on_tray action
                
                # Need to move tray if not already at child's location
                if tray_locations and child_loc != "kitchen":
                    h += 1  # move_tray action

        return h
