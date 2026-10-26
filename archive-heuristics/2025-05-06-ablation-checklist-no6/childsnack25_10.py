from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or invalid format defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The fact must have at least as many parts as the pattern args
    if len(parts) < len(args):
        return False
    # Check if the first len(args) parts match the pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts[:len(args)], args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all target children.
    The heuristic sums up the estimated costs for:
    1. Serving each unserved child.
    2. Making sandwiches that are needed but not available in the kitchen.
    3. Putting needed sandwiches onto trays.
    4. Moving trays to deliver sandwiches to children's locations.

    Assumptions:
    - Resources (bread, content, sandwich objects, trays) are sufficient if the problem is solvable.
    - Trays have sufficient capacity to carry needed sandwiches for a location.
    - put_on_tray action happens in the kitchen.
    - serve_sandwich action happens at the child's waiting location.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about
        children's waiting locations and allergy status, and identifying
        the children who need to be served from the goal state.
        """
        self.goals = task.goals  # Goal conditions (e.g., (served child1))

        # Extract static information from task.static
        self.waiting_locations = {}
        self.allergic_children = set()
        self.all_places = {'kitchen'} # Start with kitchen constant

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts
            predicate = parts[0]
            if predicate == 'waiting':
                child, place = parts[1], parts[2]
                self.waiting_locations[child] = place
                self.all_places.add(place)
            elif predicate == 'allergic_gluten':
                self.allergic_children.add(parts[1])

        # Identify the specific children who need to be served based on the goal
        self.target_children = {get_parts(g)[1] for g in self.goals if match(g, 'served', '*')}


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        # 1. Identify unserved children
        unserved_children_info = [] # List of (child, location, is_allergic)
        for child in self.target_children:
            if f'(served {child})' not in state:
                # Child is unserved, get their location and allergy status
                location = self.waiting_locations.get(child) # Should always exist for target children
                is_allergic = child in self.allergic_children
                if location: # Ensure location is known
                    unserved_children_info.append((child, location, is_allergic))

        n_unserved = len(unserved_children_info)

        # If no children are unserved, the goal is reached
        if n_unserved == 0:
            return 0

        # Heuristic starts with the number of serve actions needed
        # Each unserved child needs one serve action eventually.
        h = n_unserved

        # 2. Calculate sandwiches needed at each location
        needed_at_p = {p: {'reg': 0, 'gf': 0} for p in self.all_places}
        for _, location, is_allergic in unserved_children_info:
            if location in needed_at_p: # Ensure location is one we track
                if is_allergic:
                    needed_at_p[location]['gf'] += 1
                else:
                    needed_at_p[location]['reg'] += 1

        # 3. Calculate available suitable sandwiches on trays at each location
        avail_on_trays_at_p = {p: {'reg': 0, 'gf': 0} for p in self.all_places}
        tray_locations = {get_parts(f)[1]: get_parts(f)[2] for f in state if match(f, 'at', '*', '*') and get_parts(f)[1].startswith('tray')}

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'ontray':
                s, t = parts[1], parts[2]
                if t in tray_locations:
                    p = tray_locations[t]
                    if p in avail_on_trays_at_p: # Ensure location is one we track
                        is_gf_sandwich = f'(no_gluten_sandwich {s})' in state
                        avail_on_trays_at_p[p]['gf' if is_gf_sandwich else 'reg'] += 1

        # 4. Calculate sandwiches still needed for delivery at each location
        # These are the sandwiches needed at a location that are NOT already on a tray there.
        still_needed_at_p = {p: {'reg': 0, 'gf': 0} for p in self.all_places}
        total_still_needed_reg = 0
        total_still_needed_gf = 0

        for p in self.all_places:
             still_needed_at_p[p]['reg'] = max(0, needed_at_p[p]['reg'] - avail_on_trays_at_p[p]['reg'])
             still_needed_at_p[p]['gf'] = max(0, needed_at_p[p]['gf'] - avail_on_trays_at_p[p]['gf'])
             total_still_needed_reg += still_needed_at_p[p]['reg']
             total_still_needed_gf += still_needed_at_p[p]['gf']

        # 5. Calculate available sandwiches in the kitchen
        avail_kitchen_reg = 0
        avail_kitchen_gf = 0
        for fact in state:
            if match(fact, 'at_kitchen_sandwich', '*'):
                s = get_parts(fact)[1]
                is_gf_sandwich = f'(no_gluten_sandwich {s})' in state
                if is_gf_sandwich:
                    avail_kitchen_gf += 1
                else:
                    avail_kitchen_reg += 1

        # 6. Calculate sandwiches that must be made
        # These are needed sandwiches that are not available in the kitchen (or on trays at locations)
        # Note: total_still_needed already excludes sandwiches on trays at locations.
        must_make_reg = max(0, total_still_needed_reg - avail_kitchen_reg)
        must_make_gf = max(0, total_still_needed_gf - avail_kitchen_gf)

        h += must_make_reg + must_make_gf # Cost for make actions (one per sandwich)

        # 7. Calculate cost for putting sandwiches on trays
        # Each sandwich that needs delivery must be put on a tray in the kitchen.
        h += total_still_needed_reg + total_still_needed_gf # Cost for put_on_tray actions (one per sandwich)

        # 8. Calculate cost for tray movements
        # Identify locations that need sandwiches delivered
        locations_needing_delivery = {p for p, counts in still_needed_at_p.items() if (counts['reg'] > 0 or counts['gf'] > 0) and p != 'kitchen'} # Kitchen doesn't need delivery

        # Identify locations that currently have a tray
        locations_with_trays = set(tray_locations.values())

        # Count locations needing delivery that already have a tray (need round trip)
        locations_with_tray_needing_refill = locations_needing_delivery.intersection(locations_with_trays)

        # Count locations needing delivery that do not have a tray (need one-way trip)
        locations_without_tray_needing_delivery = locations_needing_delivery - locations_with_tray_needing_refill

        h += 2 * len(locations_with_tray_needing_refill) # Cost for round trips (move_tray kitchen->P + move_tray P->kitchen)
        h += 1 * len(locations_without_tray_needing_delivery) # Cost for one-way trips (move_tray kitchen->P)

        return h
