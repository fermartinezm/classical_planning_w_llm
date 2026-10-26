import fnmatch
import collections
import copy
from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts
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
    # Check if the number of parts matches the pattern length
    if len(parts) != len(args):
        return False
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'childsnacks'.

    # Summary
    This heuristic estimates the number of actions required to reach a state where all goal children are served.
    It calculates the cost by summing estimates for the necessary 'make_sandwich', 'put_on_tray', 'serve_sandwich',
    and 'move_tray' actions based on the current state and the unmet goals. It prioritizes using existing
    resources (sandwiches on trays or in the kitchen) before estimating the cost of making new ones.
    The move cost is estimated by counting the number of distinct locations that require a tray delivery.

    # Assumptions
    - Each unserved child requires one sandwich.
    - Allergic children require gluten-free sandwiches. Non-allergic children can eat either regular or gluten-free sandwiches.
    - Making a sandwich requires one bread portion and one content portion.
    - Putting a sandwich on a tray requires the sandwich and a tray to be in the kitchen.
    - Serving requires the correct sandwich type on a tray at the child's location.
    - Moving a tray costs 1 action per move between locations.
    - The heuristic assumes sufficient raw ingredients (bread, content) and sandwich objects ('notexist') are available if sandwiches need to be made. It doesn't check for dead ends due to resource exhaustion.
    - Trays have unlimited capacity.

    # Heuristic Initialization
    - Stores the set of children that need to be served according to the goal description.
    - Parses static facts to store each child's waiting location and allergy status (gluten-free or not).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse State:** Identify currently served children, locations of trays, sandwiches on trays (and their type - gluten-free or regular), sandwiches in the kitchen (and their type).
    2.  **Identify Unserved Children:** Determine which goal children are not yet served in the current state. If all are served, the heuristic value is 0.
    3.  **Count Needed Sandwiches:** Count how many gluten-free ('gf') sandwiches and how many regular-or-gluten-free ('regular_or_gf') sandwiches are needed for the unserved children based on their allergies.
    4.  **Count Available Sandwiches:** Count currently available sandwiches on trays and in the kitchen, categorized by type ('gf' or 'regular').
    5.  **Match Available to Needed:** Greedily assign available sandwiches to meet the needs. Prioritize using sandwiches already on trays, then those in the kitchen. Prioritize satisfying 'gf' needs first. Keep track of how many needs are met by tray sandwiches (`served_by_tray`) and kitchen sandwiches (`served_by_kitchen`).
    6.  **Calculate Make/Put/Serve Actions:**
        - `sandwiches_to_make`: Number of remaining needs after using available sandwiches.
        - `sandwiches_to_put`: Number of kitchen sandwiches used + number of sandwiches to make.
        - `sandwiches_to_serve`: Total number of unserved children.
        - Add these counts to the heuristic value `h`.
    7.  **Estimate Move Actions:**
        - Determine how many deliveries are needed at each location for the unserved children.
        - Determine how many suitable sandwiches are already on trays at those specific locations.
        - Greedily match sandwiches on trays at the target locations to the deliveries needed there. Count how many deliveries are satisfied without needing a move (`satisfied_without_move`).
        - Calculate `total_deliveries = number of unserved children`.
        - Calculate `deliveries_requiring_move = total_deliveries - satisfied_without_move`.
        - If `deliveries_requiring_move > 0`, identify the set of unique locations that still require deliveries after the local matching. The number of these locations is the estimated move cost. Add this count to `h`.
    8.  **Return Total Heuristic Value:** The final value `h` is the sum of estimated make, put, serve, and move actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        # Extract goal children
        self.goal_children = set()
        for goal_fact in self.goals:
            if match(goal_fact, "served", "*"):
                self.goal_children.add(get_parts(goal_fact)[1])

        # Process static facts
        self.child_locations = {} # child -> place
        self.child_allergies = {} # child -> bool (True if allergic)
        for fact in self.static_facts:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                self.child_locations[parts[1]] = parts[2]
            elif match(fact, "allergic_gluten", "*"):
                self.child_allergies[get_parts(fact)[1]] = True
            elif match(fact, "not_allergic_gluten", "*"):
                self.child_allergies[get_parts(fact)[1]] = False
            # We don't strictly need gf_bread/content sets unless checking solvability

        # Ensure all goal children have static info (robustness)
        for child in self.goal_children:
            if child not in self.child_locations:
                # This case should not happen in valid problems
                print(f"Warning: Child {child} from goal not found in static 'waiting' facts.")
                # Assign a default or handle error appropriately
            if child not in self.child_allergies:
                 # This case should not happen in valid problems
                print(f"Warning: Child {child} from goal not found in static allergy facts.")
                # Assign a default (e.g., not allergic) or handle error
                self.child_allergies[child] = False


    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given state node."""
        state = node.state
        h = 0

        # 1. Parse State
        served_children = set()
        kitchen_sandwiches = {'gf': set(), 'regular': set()}
        sandwiches_on_tray = {} # sandwich -> tray
        tray_locations = {} # tray -> place
        gf_sandwiches = set() # set of sandwiches that are gluten-free

        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])
            elif match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                # Check if the first argument is a tray (heuristic assumes types are known)
                # A better approach would be to get object types from the task instance
                if 'tray' in parts[1]: # Simple check based on object name convention
                     tray_locations[parts[1]] = parts[2]
            elif match(fact, "ontray", "*", "*"):
                parts = get_parts(fact)
                sandwiches_on_tray[parts[1]] = parts[2]
            elif match(fact, "at_kitchen_sandwich", "*"):
                # Type (gf/regular) determined later by checking no_gluten_sandwich
                pass # Handled below when checking no_gluten_sandwich
            elif match(fact, "no_gluten_sandwich", "*"):
                gf_sandwiches.add(get_parts(fact)[1])

        # Categorize kitchen sandwiches
        for fact in state:
             if match(fact, "at_kitchen_sandwich", "*"):
                s = get_parts(fact)[1]
                if s in gf_sandwiches:
                    kitchen_sandwiches['gf'].add(s)
                else:
                    kitchen_sandwiches['regular'].add(s)

        # 2. Identify Unserved Children
        unserved_children = self.goal_children - served_children
        if not unserved_children:
            return 0

        # 3. Count Needed Sandwiches
        needed = {'gf': 0, 'regular_or_gf': 0}
        unserved_details = [] # Store details for move calculation
        for c in unserved_children:
            is_allergic = self.child_allergies.get(c, False) # Default to false if missing
            location = self.child_locations.get(c, None) # Handle missing location if necessary
            if location is None: continue # Skip child if location unknown

            unserved_details.append({'child': c, 'location': location, 'needs_gf': is_allergic})
            if is_allergic:
                needed['gf'] += 1
            else:
                needed['regular_or_gf'] += 1

        # 4. Count Available Sandwiches
        avail_tray = {'gf': 0, 'regular': 0}
        avail_kitchen = {'gf': len(kitchen_sandwiches['gf']), 'regular': len(kitchen_sandwiches['regular'])}
        sandwiches_on_tray_at_loc = collections.defaultdict(lambda: {'gf': 0, 'regular': 0}) # loc -> type -> count

        for s, t in sandwiches_on_tray.items():
            loc = tray_locations.get(t)
            if loc is None: continue # Tray location unknown

            if s in gf_sandwiches:
                avail_tray['gf'] += 1
                sandwiches_on_tray_at_loc[loc]['gf'] += 1
            else:
                avail_tray['regular'] += 1
                sandwiches_on_tray_at_loc[loc]['regular'] += 1

        # Make copies for modification during matching
        needed_copy = copy.deepcopy(needed)
        avail_tray_copy = copy.deepcopy(avail_tray)
        avail_kitchen_copy = copy.deepcopy(avail_kitchen)

        # 5. Match Available to Needed (Greedy)
        served_by_tray = {'gf': 0, 'regular': 0}
        served_by_kitchen = {'gf': 0, 'regular': 0}

        # Satisfy GF needs first
        # Use tray GF
        count = min(needed_copy['gf'], avail_tray_copy['gf'])
        served_by_tray['gf'] += count
        needed_copy['gf'] -= count
        avail_tray_copy['gf'] -= count
        # Use kitchen GF
        count = min(needed_copy['gf'], avail_kitchen_copy['gf'])
        served_by_kitchen['gf'] += count
        needed_copy['gf'] -= count
        avail_kitchen_copy['gf'] -= count

        # Satisfy Regular_or_GF needs
        # Use tray Regular
        count = min(needed_copy['regular_or_gf'], avail_tray_copy['regular'])
        served_by_tray['regular'] += count
        needed_copy['regular_or_gf'] -= count
        avail_tray_copy['regular'] -= count
        # Use tray GF (remaining)
        count = min(needed_copy['regular_or_gf'], avail_tray_copy['gf'])
        served_by_tray['gf'] += count # Counted as GF sandwich used
        needed_copy['regular_or_gf'] -= count
        avail_tray_copy['gf'] -= count
        # Use kitchen Regular
        count = min(needed_copy['regular_or_gf'], avail_kitchen_copy['regular'])
        served_by_kitchen['regular'] += count
        needed_copy['regular_or_gf'] -= count
        avail_kitchen_copy['regular'] -= count
        # Use kitchen GF (remaining)
        count = min(needed_copy['regular_or_gf'], avail_kitchen_copy['gf'])
        served_by_kitchen['gf'] += count # Counted as GF sandwich used
        needed_copy['regular_or_gf'] -= count
        avail_kitchen_copy['gf'] -= count

        # 6. Calculate Make/Put/Serve Actions
        sandwiches_to_make = needed_copy['gf'] + needed_copy['regular_or_gf']
        sandwiches_served_by_kitchen_total = served_by_kitchen['gf'] + served_by_kitchen['regular']
        sandwiches_to_put = sandwiches_served_by_kitchen_total + sandwiches_to_make
        sandwiches_to_serve = len(unserved_children)

        h += sandwiches_to_make
        h += sandwiches_to_put
        h += sandwiches_to_serve

        # 7. Estimate Move Actions
        deliveries_needed = collections.defaultdict(lambda: {'gf': 0, 'regular_or_gf': 0})
        for detail in unserved_details:
            loc = detail['location']
            if detail['needs_gf']:
                deliveries_needed[loc]['gf'] += 1
            else:
                deliveries_needed[loc]['regular_or_gf'] += 1

        # Create a modifiable copy for matching
        sandwiches_at_loc_copy = copy.deepcopy(sandwiches_on_tray_at_loc)
        deliveries_needed_copy = copy.deepcopy(deliveries_needed)

        satisfied_without_move = 0
        locations_still_needing_deliveries = set()

        for loc, needs in deliveries_needed.items():
            available = sandwiches_at_loc_copy[loc]
            satisfied_here = 0

            # Match GF needs at this location
            count = min(needs['gf'], available['gf'])
            satisfied_here += count
            needs['gf'] -= count
            available['gf'] -= count

            # Match Regular_or_GF needs at this location
            # Use regular first
            count = min(needs['regular_or_gf'], available['regular'])
            satisfied_here += count
            needs['regular_or_gf'] -= count
            available['regular'] -= count
            # Use GF (remaining)
            count = min(needs['regular_or_gf'], available['gf'])
            satisfied_here += count
            needs['regular_or_gf'] -= count
            available['gf'] -= count

            satisfied_without_move += satisfied_here
            if needs['gf'] > 0 or needs['regular_or_gf'] > 0:
                locations_still_needing_deliveries.add(loc)

        # Add move cost based on locations still needing deliveries
        h += len(locations_still_needing_deliveries)

        # 8. Return Total Heuristic Value
        return h

