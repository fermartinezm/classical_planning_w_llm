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
    2. Accounting for sandwiches that need to be made
    3. Considering tray movements and sandwich placements
    4. Handling gluten-free requirements for allergic children

    # Assumptions:
    - Making a sandwich requires 1 action (regardless of gluten-free status)
    - Putting a sandwich on a tray requires 1 action
    - Serving a sandwich requires 1 action
    - Moving a tray between locations requires 1 action
    - We can serve multiple children at the same location with one tray movement

    # Heuristic Initialization
    - Extract information about which children are allergic to gluten
    - Identify which bread and content portions are gluten-free
    - Note all waiting children and their locations

    # Step-By-Step Thinking for Computing Heuristic
    1. Count unserved children (from goal conditions)
    2. For each unserved child:
       a. If no suitable sandwich exists:
          - If allergic: must make gluten-free sandwich (1 action)
          - Else: can make regular sandwich (1 action)
       b. If sandwich exists but not on tray: put on tray (1 action)
       c. If tray not at child's location: move tray (1 action)
       d. Serve sandwich (1 action)
    3. Optimize by:
       - Counting sandwiches that need to be made
       - Grouping children by location to minimize tray movements
       - Considering existing sandwiches on trays
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract allergic children from static facts
        self.allergic_children = {
            get_parts(fact)[1] for fact in self.static if match(fact, "allergic_gluten", "*")
        }

        # Extract gluten-free bread and content portions from static facts
        self.gluten_free_breads = {
            get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_bread", "*")
        }
        self.gluten_free_contents = {
            get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_content", "*")
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

        # If we're already in a goal state, return 0
        if self.goals <= state:
            return 0

        # Count unserved children (those in goals but not yet served)
        unserved_children = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                child = get_parts(goal)[1]
                if f"(served {child})" not in state:
                    unserved_children.add(child)

        # Track sandwiches that exist and their properties
        existing_sandwiches = set()
        gluten_free_sandwiches = set()
        sandwiches_on_trays = set()
        sandwiches_in_kitchen = set()
        tray_locations = {}  # tray -> location

        for fact in state:
            # Find existing sandwiches
            if match(fact, "notexist", "*"):
                pass  # sandwich doesn't exist
            elif match(fact, "at_kitchen_sandwich", "*"):
                sandwich = get_parts(fact)[1]
                existing_sandwiches.add(sandwich)
                sandwiches_in_kitchen.add(sandwich)
            elif match(fact, "no_gluten_sandwich", "*"):
                sandwich = get_parts(fact)[1]
                gluten_free_sandwiches.add(sandwich)
            elif match(fact, "ontray", "*", "*"):
                sandwich, tray = get_parts(fact)[1], get_parts(fact)[2]
                existing_sandwiches.add(sandwich)
                sandwiches_on_trays.add(sandwich)
            elif match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                if obj.startswith("tray"):
                    tray_locations[obj] = loc

        # Initialize cost counters
        make_sandwich_cost = 0
        put_on_tray_cost = 0
        move_tray_cost = 0
        serve_cost = len(unserved_children)  # Each child needs to be served

        # For each unserved child, determine what needs to be done
        needed_gluten_free = 0
        needed_regular = 0
        tray_movements = set()  # Locations we need to move trays to

        for child in unserved_children:
            child_location = self.waiting_children[child]
            tray_movements.add(child_location)

            # Check if child is allergic
            is_allergic = child in self.allergic_children

            # Find suitable sandwich
            suitable_sandwich = None
            if is_allergic:
                # Look for gluten-free sandwich on any tray at our location
                for sandwich in gluten_free_sandwiches:
                    if sandwich in sandwiches_on_trays:
                        # Check if tray is at child's location
                        for fact in state:
                            if match(fact, "ontray", sandwich, "*"):
                                tray = get_parts(fact)[2]
                                if tray_locations.get(tray) == child_location:
                                    suitable_sandwich = sandwich
                                    break
                        if suitable_sandwich:
                            break

                if not suitable_sandwich:
                    needed_gluten_free += 1
            else:
                # Look for any sandwich on any tray at our location
                for sandwich in existing_sandwiches:
                    if sandwich in sandwiches_on_trays:
                        # Check if tray is at child's location
                        for fact in state:
                            if match(fact, "ontray", sandwich, "*"):
                                tray = get_parts(fact)[2]
                                if tray_locations.get(tray) == child_location:
                                    suitable_sandwich = sandwich
                                    break
                        if suitable_sandwich:
                            break

                if not suitable_sandwich:
                    needed_regular += 1

        # Calculate sandwich making costs
        make_sandwich_cost = needed_gluten_free + needed_regular

        # Calculate putting sandwiches on trays
        # We need to put on tray: sandwiches that need to be made + existing ones in kitchen
        put_on_tray_cost = needed_gluten_free + needed_regular
        put_on_tray_cost += len(sandwiches_in_kitchen)

        # Calculate tray movement costs
        # We need to move trays to all locations where children are waiting
        # Subtract 1 if a tray is already at kitchen (initial position)
        kitchen_trays = sum(1 for tray, loc in tray_locations.items() if loc == "kitchen")
        move_tray_cost = max(0, len(tray_movements) - kitchen_trays)

        # Total heuristic is sum of all action costs
        total_cost = make_sandwich_cost + put_on_tray_cost + move_tray_cost + serve_cost
        return total_cost
