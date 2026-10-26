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


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved
    children. It counts the number of "missing steps" in the process pipeline
    for each unserved child's required sandwich and tray: making the sandwich,
    putting it on a tray, moving the tray to the child's location, and serving.
    It sums up the counts of items/goals waiting at each stage of this pipeline.

    # Assumptions
    - Each unserved child requires one suitable sandwich (gluten-free if allergic,
      any otherwise).
    - Each sandwich needs to be made (if not already), put on a tray, and the
      tray moved to the child's location before serving.
    - Trays can hold multiple sandwiches (infinite capacity assumed).
    - Making a sandwich requires one bread, one content, and one 'notexist'
      sandwich slot.
    - Ingredients and 'notexist' slots are shared resources.
    - Trays are shared resources and can be moved between locations.
    - The heuristic sums up the counts of items/goals needing the next step
      in the pipeline, treating them somewhat independently but considering
      resource availability limits for making sandwiches.

    # Heuristic Initialization
    - Extracts all objects (children, trays, sandwiches, bread, content, places)
      from the initial state and goals.
    - Identifies which children are allergic to gluten (static fact).
    - Stores the waiting location for each child (static fact in this domain).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1.  Identify all children who are currently not served. If none, the heuristic is 0.
    2.  For the unserved children, determine how many need gluten-free sandwiches
        and how many need regular sandwiches. Calculate the total number of
        sandwiches needed.
    3.  Count the number of sandwiches that are already made (either at the kitchen
        or on a tray), distinguishing between gluten-free and regular.
    4.  Count the number of sandwiches that are currently on trays.
    5.  Count the available ingredients (bread and content, distinguishing gluten-free
        and regular) and available 'notexist' sandwich slots in the kitchen.
    6.  Identify the locations where unserved children are waiting.
    7.  Count the number of trays currently located at any of these waiting locations.
    8.  Calculate the heuristic value as the sum of estimated costs for different stages:
        a.  **Serving Cost:** Add 1 for each unserved child (each needs a 'serve' action).
        b.  **Tray Movement Cost:** Add 1 for each location where unserved children
            are waiting but no tray is currently present (each such location needs
            a tray moved there).
        c.  **Put on Tray Cost:** Add 1 for each sandwich that needs to be on a tray
            but isn't yet. This is calculated as the total number of sandwiches needed
            minus those already on trays.
        d.  **Make Sandwich Cost:** Add 1 for each sandwich that needs to be made.
            This is calculated as the total number of sandwiches needed minus those
            already made. This count is capped by the number of sandwiches that can
            actually be made given the available ingredients and 'notexist' slots.

    The total heuristic is the sum of costs from steps 8a + 8b + 8c + 8d.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information and objects."""
        self.goals = task.goals
        self.static_facts = task.static

        # Extract objects by type from initial state, goals, and static facts
        self.all_children = set()
        self.all_trays = set()
        self.all_sandwiches = set()
        self.all_breads = set()
        self.all_contents = set()
        self.all_places = {'kitchen'} # Add constant kitchen

        all_facts = set(task.initial_state) | set(task.goals) | set(task.static)

        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any
            predicate = parts[0]
            args = parts[1:]

            if predicate in ['at_kitchen_bread', 'no_gluten_bread']:
                if args: self.all_breads.add(args[0])
            elif predicate in ['at_kitchen_content', 'no_gluten_content']:
                if args: self.all_contents.add(args[0])
            elif predicate in ['at_kitchen_sandwich', 'no_gluten_sandwich', 'notexist']:
                 if args: self.all_sandwiches.add(args[0])
            elif predicate == 'ontray':
                 if len(args) > 1:
                     self.all_sandwiches.add(args[0])
                     self.all_trays.add(args[1])
            elif predicate == 'at':
                 if len(args) > 1:
                     self.all_trays.add(args[0])
                     self.all_places.add(args[1])
            elif predicate in ['allergic_gluten', 'not_allergic_gluten', 'served']:
                 if args: self.all_children.add(args[0])
            elif predicate == 'waiting':
                 if len(args) > 1:
                     self.all_children.add(args[0])
                     self.all_places.add(args[1])

        # Store static information
        self.allergic_children = {c for c in self.all_children if '(allergic_gluten ' + c + ')' in self.static_facts}
        self.not_allergic_children = {c for c in self.all_children if '(not_allergic_gluten ' + c + ')' in self.static_facts}
        # Waiting locations are static in this domain
        self.waiting_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in task.initial_state if match(fact, 'waiting', '*', '*')}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children and their needs/locations
        unserved_children = {c for c in self.all_children if '(served ' + c + ')' not in state}
        if not unserved_children:
            return 0 # Goal state

        needed_ng = sum(1 for c in unserved_children if c in self.allergic_children)
        needed_reg = sum(1 for c in unserved_children if c in self.not_allergic_children)
        needed_total = needed_ng + needed_reg

        needed_locations = {self.waiting_locations[c] for c in unserved_children}

        # 2. Count available resources in the current state
        avail_made_ng = sum(1 for s in self.all_sandwiches if '(no_gluten_sandwich ' + s + ')' in state and ('(at_kitchen_sandwich ' + s + ')' in state or any('(ontray ' + s + ' ' + t + ')' in state for t in self.all_trays)))
        avail_made_reg = sum(1 for s in self.all_sandwiches if '(no_gluten_sandwich ' + s + ')' not in state and ('(at_kitchen_sandwich ' + s + ')' in state or any('(ontray ' + s + ' ' + t + ')' in state for t in self.all_trays)))
        avail_ontray_sandwich = sum(1 for s in self.all_sandwiches if any('(ontray ' + s + ' ' + t + ')' in state for t in self.all_trays))

        avail_bread_ng = sum(1 for b in self.all_breads if '(at_kitchen_bread ' + b + ')' in state and '(no_gluten_bread ' + b + ')' in state)
        avail_content_ng = sum(1 for c in self.all_contents if '(at_kitchen_content ' + c + ')' in state and '(no_gluten_content ' + c + ')' in state)
        avail_bread_reg = sum(1 for b in self.all_breads if '(at_kitchen_bread ' + b + ')' in state and '(no_gluten_bread ' + b + ')' not in state)
        avail_content_reg = sum(1 for c in self.all_contents if '(at_kitchen_content ' + c + ')' in state and '(no_gluten_content ' + c + ')' not in state)
        avail_notexist_sandwich = sum(1 for s in self.all_sandwiches if '(notexist ' + s + ')' in state)

        trays_at_loc = {t: p for t in self.all_trays for p in self.all_places if '(at ' + t + ' ' + p + ')' in state}
        num_trays_at_needed_locs = sum(1 for t, p in trays_at_loc.items() if p in needed_locations)


        # 3. Calculate costs based on missing prerequisites layers

        # Cost 1: Serve action for each unserved child
        # Each unserved child needs one serve action.
        h = len(unserved_children)

        # Cost 2: Tray movement for locations needing a tray
        # For each location where unserved children are waiting, if no tray is currently there,
        # we need to move one.
        num_locs_need_tray_move = max(0, len(needed_locations) - num_trays_at_needed_locs)
        h += num_locs_need_tray_move

        # Cost 3: Put on tray action for sandwiches that need to get onto trays
        # We need needed_total sandwiches to eventually be on trays.
        # avail_ontray_sandwich are already on trays.
        # The rest need to be put on trays. Each needs a put_on_tray action.
        cost_put_on_tray = max(0, needed_total - avail_ontray_sandwich)
        h += cost_put_on_tray

        # Cost 4: Make sandwich action for sandwiches that need to be made
        # We need needed_total sandwiches in total.
        # avail_made_ng + avail_made_reg are already made.
        # The rest need to be made. Each needs a make action.
        to_make = max(0, needed_total - (avail_made_ng + avail_made_reg))

        # Check ingredient/slot capacity for making
        avail_bread_total = avail_bread_ng + avail_bread_reg
        avail_content_total = avail_content_ng + avail_content_reg
        max_possible_make = min(avail_bread_total, avail_content_total, avail_notexist_sandwich)
        # Cap the number of sandwiches that can be made by the available resources
        cost_make = min(to_make, max_possible_make)
        h += cost_make

        return h

