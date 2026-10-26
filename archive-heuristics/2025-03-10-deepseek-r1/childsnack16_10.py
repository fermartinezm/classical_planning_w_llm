from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """Check if a PDDL fact matches a pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class Childsnack16Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Childsnack domain.

    # Summary
    Estimates the number of actions needed to serve all children by considering steps to prepare, move, and serve sandwiches. Each unserved child contributes to the heuristic based on their requirements (gluten-free or not) and current resource availability.

    # Assumptions
    - Each child requires a separate sandwich.
    - Gluten-free children need specific bread and content.
    - Trays start at the kitchen and can be moved between locations.
    - Sandwiches not on trays must be placed on trays at the kitchen.

    # Heuristic Initialization
    - Extracts static info: allergic children, waiting locations, gluten-free ingredients.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved child:
        a. Check if a suitable sandwich is already on a tray at their location.
        b. If not, check for suitable sandwiches on other trays, requiring movement.
        c. If no such sandwiches, check kitchen for suitable sandwiches to place on trays.
        d. If none, account for making a new sandwich and handling tray logistics.
    2. Sum actions for all unserved children, considering each child's specific needs.
    """

    def __init__(self, task):
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.child_waiting = {}
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif parts[0] == "not_allergic_gluten":
                self.not_allergic_children.add(parts[1])
            elif parts[0] == "waiting":
                self.child_waiting[parts[1]] = parts[2]
            elif parts[0] == "no_gluten_bread":
                self.gluten_free_breads.add(parts[1])
            elif parts[0] == "no_gluten_content":
                self.gluten_free_contents.add(parts[1])

    def __call__(self, node):
        state = node.state
        served = {parts[1] for fact in state if match(fact, "served", "*") for parts in [get_parts(fact)]}
        heuristic_value = 0

        for child in self.child_waiting:
            if child in served:
                continue

            is_allergic = child in self.allergic_children
            p = self.child_waiting[child]
            found = False

            # Check for suitable sandwich on tray at p
            for ontray_fact in state:
                if not match(ontray_fact, "ontray", "*", "*"):
                    continue
                s, t = get_parts(ontray_fact)[1], get_parts(ontray_fact)[2]
                tray_at_p = any(match(f, "at", t, p) for f in state)
                if not tray_at_p:
                    continue
                if is_allergic and not any(match(f, "no_gluten_sandwich", s) for f in state):
                    continue
                heuristic_value += 1
                found = True
                break
            if found:
                continue

            # Check for suitable sandwich on any tray
            has_tray_sandwich = False
            for ontray_fact in state:
                s = get_parts(ontray_fact)[1]
                if is_allergic and not any(match(f, "no_gluten_sandwich", s) for f in state):
                    continue
                has_tray_sandwich = True
                break
            if has_tray_sandwich:
                heuristic_value += 2
                continue

            # Check kitchen for suitable sandwich
            kitchen_sandwich = False
            for fact in state:
                if match(fact, "at_kitchen_sandwich", "*"):
                    s = get_parts(fact)[1]
                    if is_allergic and not any(match(f, "no_gluten_sandwich", s) for f in state):
                        continue
                    kitchen_sandwich = True
                    break
            if kitchen_sandwich:
                trays_at_kitchen = any(match(fact, "at", "tray*", "kitchen") for fact in state)
                heuristic_value += 3 if trays_at_kitchen else 4
                continue

            # Make new sandwich
            trays_at_kitchen = any(match(fact, "at", "tray*", "kitchen") for fact in state)
            heuristic_value += 4 if trays_at_kitchen else 5

        return heuristic_value
