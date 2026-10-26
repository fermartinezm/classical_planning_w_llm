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
    - The number of unserved children
    - Whether suitable sandwiches are available or need to be made
    - Whether sandwiches need to be moved to trays and trays to children

    # Assumptions:
    - Making a sandwich takes 1 action
    - Putting a sandwich on a tray takes 1 action
    - Serving a sandwich takes 1 action
    - Moving a tray between locations takes 1 action
    - We prioritize serving children with gluten allergies first since they have stricter requirements

    # Heuristic Initialization
    - Extract information about which children are allergic to gluten
    - Extract which bread and content portions are gluten-free
    - Extract goal conditions (which children need to be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (from goals not yet satisfied)
    2. For each unserved child:
       a. If allergic to gluten:
          - Check if a gluten-free sandwich is already on a tray at their location
          - If not, check if one can be made from available gluten-free ingredients
          - Add appropriate costs for making, tray placement, and moving if needed
       b. If not allergic:
          - Check if any sandwich is already on a tray at their location
          - If not, check if one can be made from available ingredients
          - Add appropriate costs
    3. For shared resources (trays, ingredients), we make optimistic assumptions
       about parallel usage to avoid overcounting
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract children's allergy information from static facts
        self.allergic_children = set()
        self.normal_children = set()
        for fact in self.static:
            if match(fact, "allergic_gluten", "*"):
                self.allergic_children.add(get_parts(fact)[1])
            elif match(fact, "not_allergic_gluten", "*"):
                self.normal_children.add(get_parts(fact)[1])

        # Extract gluten-free ingredients from static facts
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()
        for fact in self.static:
            if match(fact, "no_gluten_bread", "*"):
                self.gluten_free_breads.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_content", "*"):
                self.gluten_free_contents.add(get_parts(fact)[1])

    def __call__(self, node):
        """Estimate the minimum cost to serve all children."""
        state = node.state
        total_cost = 0

        # Find which children still need to be served
        unserved_children = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                child = get_parts(goal)[1]
                if f"(served {child})" not in state:
                    unserved_children.add(child)

        if not unserved_children:
            return 0  # Goal state

        # Extract current state information
        waiting_children = {}  # child: place
        sandwiches_on_trays = {}  # sandwich: tray
        tray_locations = {}  # tray: place
        kitchen_sandwiches = set()
        kitchen_breads = set()
        kitchen_contents = set()
        gluten_free_sandwiches = set()

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "waiting", "*", "*"):
                waiting_children[parts[1]] = parts[2]
            elif match(fact, "ontray", "*", "*"):
                sandwiches_on_trays[parts[1]] = parts[2]
            elif match(fact, "at", "*", "*"):
                if parts[1].startswith("tray"):
                    tray_locations[parts[1]] = parts[2]
            elif match(fact, "at_kitchen_sandwich", "*"):
                kitchen_sandwiches.add(parts[1])
            elif match(fact, "at_kitchen_bread", "*"):
                kitchen_breads.add(parts[1])
            elif match(fact, "at_kitchen_content", "*"):
                kitchen_contents.add(parts[1])
            elif match(fact, "no_gluten_sandwich", "*"):
                gluten_free_sandwiches.add(parts[1])

        # Process allergic children first since they have stricter requirements
        for child in unserved_children:
            if child in self.allergic_children:
                # Check if there's already a gluten-free sandwich at child's location
                served = False
                for sandwich, tray in sandwiches_on_trays.items():
                    if sandwich in gluten_free_sandwiches and tray_locations.get(tray, None) == waiting_children[child]:
                        served = True
                        break
                
                if served:
                    continue

                # Check if we can make a gluten-free sandwich
                can_make = False
                for bread in kitchen_breads:
                    if bread in self.gluten_free_breads:
                        for content in kitchen_contents:
                            if content in self.gluten_free_contents:
                                can_make = True
                                break
                        if can_make:
                            break

                if can_make:
                    total_cost += 1  # make_sandwich_no_gluten
                    # Assume we can put it on a tray and move it in one step
                    total_cost += 1  # put_on_tray
                    if tray_locations.get(tray, None) != waiting_children[child]:
                        total_cost += 1  # move_tray
                    total_cost += 1  # serve_sandwich_no_gluten
                else:
                    # No way to serve this child currently - penalize heavily
                    total_cost += 10
            else:
                # Non-allergic child - simpler requirements
                # Check if there's any sandwich at child's location
                served = False
                for sandwich, tray in sandwiches_on_trays.items():
                    if tray_locations.get(tray, None) == waiting_children[child]:
                        served = True
                        break
                
                if served:
                    continue

                # Check if we can make any sandwich
                if kitchen_breads and kitchen_contents:
                    total_cost += 1  # make_sandwich
                    total_cost += 1  # put_on_tray
                    if tray_locations.get(tray, None) != waiting_children[child]:
                        total_cost += 1  # move_tray
                    total_cost += 1  # serve_sandwich
                else:
                    # No ingredients left - penalize
                    total_cost += 10

        return total_cost
