from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnack17Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Childsnack domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children by considering:
    - Current locations of sandwiches and trays.
    - Availability of gluten-free ingredients for allergic children.
    - Steps needed to make, move, and serve sandwiches.

    # Assumptions
    - Allergic children require gluten-free sandwiches.
    - Non-allergic children can be served any sandwich.
    - Trays can be moved between places in one action.
    - Making a sandwich consumes one bread and one content portion.

    # Heuristic Initialization
    - Extracts static information about allergies and gluten-free ingredients.
    - Collects all objects (children, breads, contents, sandwiches, trays, places) from the initial state and static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved child:
        a. Determine if they are allergic.
        b. Check existing suitable sandwiches on trays or in the kitchen.
        c. Calculate the minimal steps to serve them, considering:
            - Moving trays.
            - Making new sandwiches if needed.
    2. Sum the minimal steps for all unserved children.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract all objects from initial state and static facts
        self.children = set()
        self.breads = set()
        self.contents = set()
        self.sandwiches = set()
        self.trays = set()
        self.places = {'kitchen'}

        # Extract from static and initial state
        for fact in task.static | task.initial_state:
            parts = get_parts(fact)
            if parts[0] in ['allergic_gluten', 'not_allergic_gluten']:
                self.children.add(parts[1])
            elif parts[0] in ['at_kitchen_bread', 'no_gluten_bread']:
                self.breads.add(parts[1])
            elif parts[0] in ['at_kitchen_content', 'no_gluten_content']:
                self.contents.add(parts[1])
            elif parts[0] in ['notexist', 'at_kitchen_sandwich', 'no_gluten_sandwich']:
                self.sandwiches.add(parts[1])
            elif parts[0] == 'ontray':
                self.sandwiches.add(parts[1])
                self.trays.add(parts[2])
            elif parts[0] == 'at':
                self.trays.add(parts[1])
                self.places.add(parts[2])
            elif parts[0] == 'waiting':
                self.children.add(parts[1])
                self.places.add(parts[2])

        # Determine allergic children
        self.allergic = {child: False for child in self.children}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.allergic[parts[1]] = True
            elif parts[0] == 'not_allergic_gluten':
                self.allergic[parts[1]] = False

        # Gluten-free ingredients
        self.gluten_free_bread = {b: False for b in self.breads}
        self.gluten_free_content = {c: False for c in self.contents}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'no_gluten_bread':
                self.gluten_free_bread[parts[1]] = True
            elif parts[0] == 'no_gluten_content':
                self.gluten_free_content[parts[1]] = True

    def __call__(self, node):
        state = node.state
        total_cost = 0

        served = {parts[1] for fact in state if match(fact, 'served', '*')}
        unserved = [c for c in self.children if c not in served]

        for child in unserved:
            # Get child's location
            waiting_facts = [fact for fact in state if match(fact, 'waiting', child, '*')]
            if not waiting_facts:
                continue  # Should not happen in valid states
            child_place = get_parts(waiting_facts[0])[2]

            if self.allergic[child]:
                # Need gluten-free sandwich
                gf_sandwiches = [s for s in self.sandwiches 
                                if f'(no_gluten_sandwich {s})' in state 
                                and f'(notexist {s})' not in state]
                min_cost = float('inf')

                for s in gf_sandwiches:
                    ontray_facts = [fact for fact in state if match(fact, 'ontray', s, '*')]
                    if ontray_facts:
                        tray = get_parts(ontray_facts[0])[2]
                        tray_loc_facts = [fact for fact in state if match(fact, 'at', tray, '*')]
                        if not tray_loc_facts:
                            continue
                        tray_loc = get_parts(tray_loc_facts[0])[2]
                        if tray_loc == child_place:
                            cost = 1
                        else:
                            cost = 2  # move + serve
                    else:
                        if f'(at_kitchen_sandwich {s})' in state:
                            cost = 3  # put_on_tray (1) + move (1) + serve (1)
                        else:
                            continue  # Sandwich not accessible
                    if cost < min_cost:
                        min_cost = cost

                # If no existing GF sandwiches, check if we can make one
                if not gf_sandwiches:
                    available_gf_bread = [b for b in self.breads 
                                         if f'(at_kitchen_bread {b})' in state 
                                         and self.gluten_free_bread[b]]
                    available_gf_content = [c for c in self.contents 
                                           if f'(at_kitchen_content {c})' in state 
                                           and self.gluten_free_content[c]]
                    if available_gf_bread and available_gf_content:
                        min_cost = min(min_cost, 4)  # make(1) + put(1) + move(1) + serve(1)
                    else:
                        # Assume problem is solvable, use 4 as fallback
                        min_cost = min(min_cost, 4)

                total_cost += min_cost if min_cost != float('inf') else 4
            else:
                # Can use any sandwich
                existing_sandwiches = [s for s in self.sandwiches 
                                      if f'(notexist {s})' not in state]
                min_cost = float('inf')

                for s in existing_sandwiches:
                    ontray_facts = [fact for fact in state if match(fact, 'ontray', s, '*')]
                    if ontray_facts:
                        tray = get_parts(ontray_facts[0])[2]
                        tray_loc_facts = [fact for fact in state if match(fact, 'at', tray, '*')]
                        if not tray_loc_facts:
                            continue
                        tray_loc = get_parts(tray_loc_facts[0])[2]
                        if tray_loc == child_place:
                            cost = 1
                        else:
                            cost = 2
                    else:
                        if f'(at_kitchen_sandwich {s})' in state:
                            cost = 3  # put(1) + move(1) + serve(1)
                        else:
                            continue  # Sandwich not accessible
                    if cost < min_cost:
                        min_cost = cost

                # If no existing sandwiches, make new
                if not existing_sandwiches:
                    available_bread = [b for b in self.breads 
                                      if f'(at_kitchen_bread {b})' in state]
                    available_content = [c for c in self.contents 
                                        if f'(at_kitchen_content {c})' in state]
                    if available_bread and available_content:
                        min_cost = min(min_cost, 4)
                    else:
                        min_cost = min(min_cost, 4)

                total_cost += min_cost if min_cost != float('inf') else 4

        return total_cost
