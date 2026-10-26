from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class Childsnack4Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children by considering the steps required for each unserved child to receive their appropriate sandwich. For allergic children, this involves ensuring a no-gluten sandwich is available on a tray at their location. For non-allergic children, any sandwich suffices. The heuristic sums the minimal steps required for each child, considering current sandwich and tray positions, and the availability of ingredients.

    # Assumptions:
    - Each child is waiting at a fixed location (from static facts).
    - Allergic children require a no-gluten sandwich, which must be on a tray at their location.
    - Non-allergic children can be served with any sandwich.
    - Trays can be moved between locations with one action per move.
    - Making a sandwich requires available ingredients (bread and content) at the kitchen.

    # Heuristic Initialization
    - Extract allergic status and waiting locations for each child from static facts.
    - Identify no-gluten bread and content types from static facts.
    - Collect all available bread and content types from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved child:
        a. If allergic:
            i. Check if a no-gluten sandwich is on a tray at their location. If yes, cost +=0.
            ii. Else, check if any no-gluten sandwich is on a tray elsewhere. Cost +=2 (move and serve).
            iii. Else, check if a no-gluten sandwich is in the kitchen. Cost +=3 (put on tray, move, serve).
            iv. Else, check if no-gluten ingredients are available. Cost +=4 (make, put, move, serve).
        b. If non-allergic:
            i. Check if any sandwich is on a tray at their location. If yes, cost +=0.
            ii. Else, check if any sandwich is on a tray elsewhere. Cost +=2 (move and serve).
            iii. Else, check if any sandwich is in the kitchen. Cost +=3 (put on tray, move, serve).
            iv. Else, check if any ingredients are available. Cost +=4 (make, put, move, serve).
    2. Sum the costs for all unserved children.
    """

    def __init__(self, task):
        """Initialize the heuristic with static and initial state information."""
        self.static = task.static
        self.child_info = {}
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()
        self.all_breads = set()
        self.all_contents = set()

        # Extract child information and no-gluten ingredients from static facts
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'allergic_gluten':
                child = parts[1]
                self.child_info[child] = {'allergic': True, 'location': None}
            elif parts[0] == 'not_allergic_gluten':
                child = parts[1]
                self.child_info[child] = {'allergic': False, 'location': None}
            elif parts[0] == 'waiting':
                child, location = parts[1], parts[2]
                if child in self.child_info:
                    self.child_info[child]['location'] = location
            elif parts[0] == 'no_gluten_bread':
                self.no_gluten_breads.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.no_gluten_contents.add(parts[1])

        # Extract all breads and contents from initial state
        for fact in task.initial_state:
            parts = fact[1:-1].split()
            if parts[0] == 'at_kitchen_bread':
                self.all_breads.add(parts[1])
            elif parts[0] == 'at_kitchen_content':
                self.all_contents.add(parts[1])

    def __call__(self, node):
        """Compute the heuristic estimate for the given state."""
        state = node.state
        total_cost = 0
        served = set()

        # Collect served children
        for fact in state:
            if fact.startswith('(served '):
                served.add(fact[1:-1].split()[1])

        for child, info in self.child_info.items():
            if child in served:
                continue

            is_allergic = info['allergic']
            location = info['location']

            if is_allergic:
                # Check for no-gluten sandwich on tray at location
                found = False
                for fact in state:
                    if fact.startswith('(ontray '):
                        parts = fact[1:-1].split()
                        s, tray = parts[1], parts[2]
                        # Check tray location
                        tray_loc = None
                        for f in state:
                            if f.startswith(f'(at {tray} '):
                                tray_loc = f[1:-1].split()[2]
                                break
                        if tray_loc == location and f'(no_gluten_sandwich {s})' in state:
                            found = True
                            break
                if found:
                    continue

                # Check if any no-gluten sandwich on any tray (but wrong location)
                tray_found = False
                for fact in state:
                    if fact.startswith('(ontray '):
                        s, tray = fact[1:-1].split()[1], fact[1:-1].split()[2]
                        if f'(no_gluten_sandwich {s})' in state:
                            # Check tray location
                            tray_loc = None
                            for f in state:
                                if f.startswith(f'(at {tray} '):
                                    tray_loc = f[1:-1].split()[2]
                                    break
                            if tray_loc != location:
                                tray_found = True
                                break
                if tray_found:
                    total_cost += 2  # move and serve
                    continue

                # Check if any no-gluten sandwich in kitchen
                kitchen_sandwich = False
                for fact in state:
                    if fact.startswith('(at_kitchen_sandwich '):
                        s = fact[1:-1].split()[1]
                        if f'(no_gluten_sandwich {s})' in state:
                            kitchen_sandwich = True
                            break
                if kitchen_sandwich:
                    total_cost += 3  # put, move, serve
                    continue

                # Check available no-gluten ingredients
                available_bread = any(f'(at_kitchen_bread {b})' in state for b in self.no_gluten_breads)
                available_content = any(f'(at_kitchen_content {c})' in state for c in self.no_gluten_contents)
                if available_bread and available_content:
                    total_cost += 4  # make, put, move, serve
                else:
                    # Assume ingredients are available (problem is solvable)
                    total_cost += 4
            else:
                # Non-allergic child
                # Check any sandwich on tray at location
                found = False
                for fact in state:
                    if fact.startswith('(ontray '):
                        tray = fact[1:-1].split()[2]
                        # Check tray location
                        tray_loc = None
                        for f in state:
                            if f.startswith(f'(at {tray} '):
                                tray_loc = f[1:-1].split()[2]
                                break
                        if tray_loc == location:
                            found = True
                            break
                if found:
                    continue

                # Check any sandwich on tray elsewhere
                tray_found = False
                for fact in state:
                    if fact.startswith('(ontray '):
                        tray = fact[1:-1].split()[2]
                        # Check tray location
                        tray_loc = None
                        for f in state:
                            if f.startswith(f'(at {tray} '):
                                tray_loc = f[1:-1].split()[2]
                                break
                        if tray_loc != location:
                            tray_found = True
                            break
                if tray_found:
                    total_cost += 2
                    continue

                # Check any sandwich in kitchen
                kitchen_sandwich = any(fact.startswith('(at_kitchen_sandwich ') for fact in state)
                if kitchen_sandwich:
                    total_cost += 3
                    continue

                # Check available any bread and content
                available_bread = any(f'(at_kitchen_bread {b})' in state for b in self.all_breads)
                available_content = any(f'(at_kitchen_content {c})' in state for c in self.all_contents)
                if available_bread and available_content:
                    total_cost += 4
                else:
                    # Assume available
                    total_cost += 4

        return total_cost
