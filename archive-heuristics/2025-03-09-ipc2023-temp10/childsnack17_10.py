from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnack17Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children with sandwiches.
    It considers the number of children waiting, the number of sandwiches that need to be made,
    and the number of trays available. It prioritizes making sandwiches for allergic children with
    no-gluten ingredients.

    # Assumptions
    - Each child needs one sandwich.
    - Making a sandwich requires one bread-portion and one content-portion.
    - Trays can be moved to different places.

    # Heuristic Initialization
    - Extract the number of allergic and non-allergic children, available bread and content portions (gluten-free and not),
      and the number of trays.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children waiting to be served.
    2. Count the number of gluten-allergic children waiting.
    3. Count the number of available gluten-free bread and content portions.
    4. Count the number of available bread and content portions (including gluten-free).
    5. Count the number of available trays.
    6. Count the number of sandwiches on trays.
    7. Estimate the number of "make_sandwich_no_gluten" actions needed for allergic children.
    8. Estimate the number of "make_sandwich" actions needed for non-allergic children.
    9. Estimate the number of "put_on_tray" actions needed.
    10. Estimate the number of "serve_sandwich_no_gluten" actions needed for allergic children.
    11. Estimate the number of "serve_sandwich" actions needed for non-allergic children.
    12. Estimate the number of "move_tray" actions needed.
    13. Sum up all the estimated actions to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting relevant information from the task."""
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.no_gluten_bread = set()
        self.no_gluten_content = set()
        self.waiting_children = set()
        self.trays = set()

        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                self.allergic_children.add(get_parts(fact)[1])
            elif match(fact, "not_allergic_gluten", "*"):
                self.not_allergic_children.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_bread", "*"):
                self.no_gluten_bread.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_content", "*"):
                self.no_gluten_content.add(get_parts(fact)[1])
            elif match(fact, "waiting", "*", "*"):
                self.waiting_children.add(get_parts(fact)[1])

        for fact in task.initial_state:
            if match(fact, "at", "*", "kitchen"):
                self.trays.add(get_parts(fact)[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        served_children = set()
        sandwiches_on_trays = set()
        at_kitchen_bread = set()
        at_kitchen_content = set()
        at_kitchen_sandwich = set()
        waiting = set()
        trays_at_location = {}
        notexist_sandwiches = set()

        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])
            elif match(fact, "ontray", "*", "*"):
                sandwiches_on_trays.add(get_parts(fact)[1])
            elif match(fact, "at_kitchen_bread", "*"):
                at_kitchen_bread.add(get_parts(fact)[1])
            elif match(fact, "at_kitchen_content", "*"):
                at_kitchen_content.add(get_parts(fact)[1])
            elif match(fact, "at_kitchen_sandwich", "*"):
                at_kitchen_sandwich.add(get_parts(fact)[1])
            elif match(fact, "waiting", "*", "*"):
                waiting.add(get_parts(fact)[1])
            elif match(fact, "at", "*", "*"):
                tray, location = get_parts(fact)[1], get_parts(fact)[2]
                if tray in self.trays:
                    trays_at_location[tray] = location
            elif match(fact, "notexist", "*"):
                notexist_sandwiches.add(get_parts(fact)[1])

        unserved_children = waiting - served_children
        num_unserved = len(unserved_children)

        if num_unserved == 0:
            return 0

        num_allergic_unserved = len(self.allergic_children.intersection(unserved_children))
        num_not_allergic_unserved = num_unserved - num_allergic_unserved

        num_no_gluten_bread = len(self.no_gluten_bread.intersection(at_kitchen_bread))
        num_no_gluten_content = len(self.no_gluten_content.intersection(at_kitchen_content))
        num_bread = len(at_kitchen_bread)
        num_content = len(at_kitchen_content)
        num_trays = len(self.trays)
        num_sandwiches_on_trays = len(sandwiches_on_trays)

        make_sandwich_no_gluten_actions = 0
        if num_allergic_unserved > 0:
            make_sandwich_no_gluten_actions = min(num_allergic_unserved, num_no_gluten_bread, num_no_gluten_content, len(notexist_sandwiches))

        make_sandwich_actions = 0
        if num_not_allergic_unserved > 0:
            make_sandwich_actions = min(num_not_allergic_unserved, num_bread, num_content, len(notexist_sandwiches) - make_sandwich_no_gluten_actions)

        put_on_tray_actions = min(make_sandwich_no_gluten_actions + make_sandwich_actions, num_trays - num_sandwiches_on_trays)

        serve_sandwich_no_gluten_actions = 0
        serve_sandwich_actions = 0

        for tray in self.trays:
            if tray in trays_at_location:
                location = trays_at_location[tray]
                for child in self.allergic_children:
                    if child in unserved_children:
                        for sandwich in sandwiches_on_trays:
                            if (f"(ontray {sandwich} {tray})" in state) and (f"({child} {location})" in state) and (f"(no_gluten_sandwich {sandwich})" in state):
                                serve_sandwich_no_gluten_actions += 1
                                break
                for child in self.not_allergic_children:
                    if child in unserved_children:
                        for sandwich in sandwiches_on_trays:
                            if (f"(ontray {sandwich} {tray})" in state) and (f"({child} {location})" in state) and (f"(no_gluten_sandwich {sandwich})" not in state):
                                serve_sandwich_actions += 1
                                break

        move_tray_actions = 0
        for tray in self.trays:
            if tray in trays_at_location:
                location = trays_at_location[tray]
                children_at_location = 0
                for child in unserved_children:
                    for fact in static_facts:
                        if match(fact, "waiting", child, location):
                            children_at_location += 1
                            break
                if children_at_location > 0:
                    move_tray_actions += 1

        heuristic_value = (
            make_sandwich_no_gluten_actions
            + make_sandwich_actions
            + put_on_tray_actions
            + serve_sandwich_no_gluten_actions
            + serve_sandwich_actions
            + move_tray_actions
        )

        return heuristic_value
