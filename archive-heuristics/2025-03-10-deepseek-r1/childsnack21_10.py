from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnack21Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children by considering:
    - Whether each child is allergic to gluten and requires a gluten-free sandwich.
    - The current locations of trays and sandwiches.
    - The availability of ingredients to make new sandwiches.

    # Assumptions
    - Each sandwich can only be used to serve one child.
    - Moving a tray between any two places takes one action.
    - Making a sandwich requires one action, and putting it on a tray takes another action.
    - Serving a child takes one action once the correct sandwich is on a tray at their location.

    # Heuristic Initialization
    - Extract allergic children, their waiting locations, gluten-free ingredients, and all possible sandwiches from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved child:
        a. If allergic:
            i. Check for a gluten-free sandwich on a tray at their location. If found, cost +=1.
            ii. If not found, check for any gluten-free sandwich on other trays. Cost += move tray and serve (2 actions).
            iii. If no such sandwich exists, cost += making a new gluten-free sandwich, putting it on a tray, moving the tray, and serving (4 actions).
        b. If not allergic:
            i. Check for any sandwich on a tray at their location. If found, cost +=1.
            ii. If not found, check for any sandwich on other trays. Cost += move tray and serve (2 actions).
            iii. If no sandwich exists, cost += making a new sandwich, putting it on a tray, moving the tray, and serving (4 actions).
    """

    def __init__(self, task):
        self.goals = task.goals
        static = task.static

        self.allergic_children = set()
        self.child_location = {}
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()
        self.all_sandwiches = set()

        for fact in static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'waiting':
                self.child_location[parts[1]] = parts[2]
            elif parts[0] == 'no_gluten_bread':
                self.gluten_free_breads.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.gluten_free_contents.add(parts[1])
            elif parts[0] == 'notexist' and len(parts) == 2:
                self.all_sandwiches.add(parts[1])

    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Precompute notexist sandwiches
        notexist_sandwiches = set()
        for fact in state:
            if match(fact, 'notexist', '*'):
                parts = get_parts(fact)
                notexist_sandwiches.add(parts[1])
        existing_sandwiches = [s for s in self.all_sandwiches if s not in notexist_sandwiches]

        # Collect gluten-free sandwiches
        gluten_free_sandwiches = set()
        for fact in state:
            if match(fact, 'no_gluten_sandwich', '*'):
                parts = get_parts(fact)
                gluten_free_sandwiches.add(parts[1])

        # Collect tray locations and sandwiches on trays
        tray_locations = {}
        ontray = defaultdict(list)
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1].startswith('tray'):
                tray_locations[parts[1]] = parts[2]
            elif parts[0] == 'ontray':
                ontray[parts[2]].append(parts[1])

        # Available ingredients
        available_gluten_free_breads = [b for b in self.gluten_free_breads if any(match(fact, 'at_kitchen_bread', b) for fact in state)]
        available_gluten_free_contents = [c for c in self.gluten_free_contents if any(match(fact, 'at_kitchen_content', c) for fact in state)]
        available_breads = [parts[1] for fact in state if match(fact, 'at_kitchen_bread', '*')]
        available_contents = [parts[1] for fact in state if match(fact, 'at_kitchen_content', '*')]

        for child, loc in self.child_location.items():
            if f'(served {child})' in state:
                continue

            if child in self.allergic_children:
                # Check for existing gluten-free sandwich at location
                found = False
                for tray in tray_locations:
                    if tray_locations[tray] == loc:
                        for s in ontray[tray]:
                            if s in gluten_free_sandwiches:
                                total_cost += 1
                                found = True
                                break
                        if found:
                            break
                if found:
                    continue

                # Check for any gluten-free sandwich on other trays
                min_cost = float('inf')
                for s in gluten_free_sandwiches:
                    for tray in ontray:
                        if s in ontray[tray]:
                            tray_loc = tray_locations.get(tray, None)
                            if tray_loc is None:
                                continue
                            cost = 2 if tray_loc != loc else 1
                            if cost < min_cost:
                                min_cost = cost
                if min_cost != float('inf'):
                    total_cost += min_cost
                    continue

                # Make new gluten-free sandwich
                if available_gluten_free_breads and available_gluten_free_contents:
                    total_cost += 4  # make, put, move, serve
                else:
                    total_cost += 4  # Assume ingredients are available
            else:
                # Non-allergic child
                found = False
                for tray in tray_locations:
                    if tray_locations[tray] == loc and ontray[tray]:
                        total_cost += 1
                        found = True
                        break
                if found:
                    continue

                # Check for any sandwich on other trays
                min_cost = float('inf')
                for tray in ontray:
                    if ontray[tray]:
                        tray_loc = tray_locations.get(tray, None)
                        if tray_loc is None:
                            continue
                        cost = 2 if tray_loc != loc else 1
                        if cost < min_cost:
                            min_cost = cost
                if min_cost != float('inf'):
                    total_cost += min_cost
                    continue

                # Make new sandwich
                if available_breads and available_contents:
                    total_cost += 4
                else:
                    total_cost += 4  # Assume ingredients are available

        return total_cost
