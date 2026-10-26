from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact.strip('()').split()

def match(fact, pattern):
    """Check if a PDDL fact matches a given pattern using wildcards."""
    fact_parts = get_parts(fact)
    pattern_parts = pattern.split()
    if len(fact_parts) != len(pattern_parts):
        return False
    return all(fnmatch(fact_part, pattern_part) for fact_part, pattern_part in zip(fact_parts, pattern_parts))

class childsnack7Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    Estimates the number of actions required to serve all children by considering:
    - Whether each child is allergic to gluten.
    - Availability of gluten-free and regular sandwiches.
    - Current locations of trays and sandwiches.

    # Assumptions
    - Each sandwich can only be used once (served to one child).
    - Trays can be moved between locations in one action.
    - Sufficient resources (bread, content) are available to make required sandwiches.

    # Heuristic Initialization
    - Extracts allergic and non-allergic children from static facts.
    - Identifies goal children from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved child:
        a. If allergic:
            i. Check for a gluten-free sandwich on a tray at their location (1 action).
            ii. If not, check for a gluten-free sandwich in the kitchen (3 actions: put, move, serve).
            iii. If no kitchen sandwich, check trays elsewhere (2 actions: move, serve).
            iv. If none, make a new sandwich (4 actions: make, put, move, serve).
        b. If non-allergic:
            i. Check for any sandwich on a tray at their location (1 action).
            ii. If not, check kitchen for any sandwich (3 actions: put, move, serve).
            iii. If no kitchen sandwich, check trays elsewhere (2 actions: move, serve).
            iv. If none, make a new sandwich (4 actions: make, put, move, serve).
    2. Sum the estimated actions for all unserved children.
    """

    def __init__(self, task):
        """Initialize with static info and goals."""
        self.allergic_children = set()
        self.non_allergic_children = set()
        for fact in task.static:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif len(parts) == 2 and parts[0] == 'not_allergic_gluten':
                self.non_allergic_children.add(parts[1])
        
        self.goal_children = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if len(parts) == 2 and parts[0] == 'served':
                self.goal_children.add(parts[1])

    def __call__(self, node):
        """Compute heuristic estimate for the current state."""
        state = node.state
        served = set()
        waiting = {}
        sandwiches_in_kitchen = set()
        gluten_free_sandwiches = set()
        sandwiches_on_trays = {}
        tray_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == 'served' and len(parts) == 2:
                served.add(parts[1])
            elif parts[0] == 'waiting' and len(parts) == 3:
                waiting[parts[1]] = parts[2]
            elif parts[0] == 'at_kitchen_sandwich' and len(parts) == 2:
                sandwiches_in_kitchen.add(parts[1])
            elif parts[0] == 'no_gluten_sandwich' and len(parts) == 2:
                gluten_free_sandwiches.add(parts[1])
            elif parts[0] == 'ontray' and len(parts) == 3:
                sandwiches_on_trays[parts[1]] = parts[2]
            elif parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('tray'):
                tray_locations[parts[1]] = parts[2]

        h = 0
        for child in self.goal_children:
            if child in served:
                continue
            if child not in waiting:
                continue  # Should not happen in valid states
            p = waiting[child]
            
            if child in self.allergic_children:
                # Check for gluten-free sandwich on tray at p
                found = False
                for s in gluten_free_sandwiches:
                    tray = sandwiches_on_trays.get(s)
                    if tray and tray_locations.get(tray) == p:
                        found = True
                        break
                if found:
                    h += 1
                    continue
                
                # Check kitchen for gluten-free sandwich
                kitchen_available = any(s in sandwiches_in_kitchen for s in gluten_free_sandwiches)
                if kitchen_available:
                    trays_in_kitchen = [t for t, loc in tray_locations.items() if loc == 'kitchen']
                    if trays_in_kitchen:
                        h += 3  # put_on_tray (1), move (1), serve (1)
                    else:
                        h += 4  # move tray to kitchen (1) + above
                else:
                    # Check other trays
                    found_elsewhere = False
                    for s in gluten_free_sandwiches:
                        tray = sandwiches_on_trays.get(s)
                        if tray and tray_locations.get(tray) != p:
                            h += 2  # move (1), serve (1)
                            found_elsewhere = True
                            break
                    if not found_elsewhere:
                        h += 4  # make (1), put (1), move (1), serve (1)
            else:
                # Non-allergic: any sandwich
                found = False
                for s in sandwiches_on_trays:
                    tray = sandwiches_on_trays[s]
                    if tray_locations.get(tray) == p:
                        found = True
                        break
                if found:
                    h += 1
                    continue
                
                if sandwiches_in_kitchen:
                    trays_in_kitchen = [t for t, loc in tray_locations.items() if loc == 'kitchen']
                    if trays_in_kitchen:
                        h += 3  # put (1), move (1), serve (1)
                    else:
                        h += 4  # move tray to kitchen (1) + above
                else:
                    if sandwiches_on_trays:
                        h += 2  # move (1), serve (1)
                    else:
                        h += 4  # make (1), put (1), move (1), serve (1)
        return h
