from fnmatch import fnmatch
# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not running in the planner environment
# In a real environment, this should be removed and the import above uncommented.
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Access initial state for object parsing

    def __call__(self, node):
        raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The fact must have at least as many parts as the pattern arguments
    if len(parts) < len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting
    children. It calculates the cost based on the current state of suitable
    sandwiches (gluten-free for allergic children, any for others) by summing
    the estimated actions needed to move each required sandwich through a
    pipeline of stages: not-exist -> kitchen -> on-tray-kitchen -> on-tray-table -> served.
    It prioritizes satisfying allergic children first using gluten-free sandwiches.

    # Assumptions
    - Each waiting child requires exactly one sandwich.
    - Gluten-free sandwiches require both gluten-free bread and gluten-free content.
    - Regular (gluten) sandwiches can be made from any other combination of bread/content.
    - The number of makable sandwiches of each type is limited by the available ingredients in the kitchen.
    - Tray capacity is sufficient to carry needed sandwiches for a table trip.
    - Tray movements between kitchen and tables cost 1 action.
    - Putting a sandwich on a tray in the kitchen costs 1 action.
    - Making a sandwich in the kitchen costs 1 action.
    - Serving a sandwich from a tray at a table costs 1 action.
    - The heuristic does not explicitly model tray availability or location beyond kitchen/table distinction for sandwiches on trays.

    # Heuristic Initialization
    - Identify the set of children that need to be served from the task goals.
    - Extract allergy information for each child from static facts.
    - Extract gluten-free information for bread and content types from static facts.
    - Identify the set of all potential sandwich objects from the initial state facts.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify all children who are currently waiting and not yet served. Separate them into allergic and non-allergic groups. Count the number of sandwiches needed for each group (`num_ng_needed`, `num_any_needed`).
    2.  Count the available bread and content portions in the kitchen, distinguishing between gluten-free and regular types, using static facts about gluten status.
    3.  Calculate the number of gluten-free and regular sandwiches that can potentially be made from the available ingredients (`num_makable_ng`, `num_makable_g`). Assume a gluten-free sandwich requires both GF bread and GF content, and a regular sandwich can be made from any other combination, limited by total ingredients.
    4.  Categorize all potential sandwich objects based on their current state:
        - `at_table`: on a tray that is at a table.
        - `on_tray_kitchen`: on a tray that is in the kitchen.
        - `kitchen`: in the kitchen, not on a tray (`at_kitchen_sandwich`).
        - `notexist`: not yet made (`notexist`).
        Distinguish between gluten-free and regular sandwiches in each category based on the `no_gluten_sandwich` predicate in the current state. For `notexist` sandwiches, use the makable counts calculated in step 3.
    5.  Calculate the cost to serve the needed sandwiches using a layered approach, prioritizing the more constrained need (gluten-free for allergic children) and the sandwiches closest to being served.
    6.  For `num_ng_needed` sandwiches:
        - Use available NG sandwiches `at_table` first (cost 1 per sandwich).
        - Then use available NG sandwiches `on_tray_kitchen` (cost 2 per sandwich: move tray + serve).
        - Then use available NG sandwiches `kitchen` (cost 3 per sandwich: put on tray + move tray + serve).
        - Then use available NG sandwiches `notexist` (makable) (cost 4 per sandwich: make + put on tray + move tray + serve).
    7.  For `num_any_needed` sandwiches:
        - Use remaining available Any sandwiches (NG or G) `at_table` first (cost 1 per sandwich).
        - Then use remaining available Any sandwiches `on_tray_kitchen` (cost 2 per sandwich).
        - Then use remaining available Any sandwiches `kitchen` (cost 3 per sandwich).
        - Then use remaining available Any sandwiches `notexist` (makable) (cost 4 per sandwich).
    8.  Sum the costs calculated in steps 6 and 7. This total is the heuristic value.
    9.  If no children are waiting to be served, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children, allergy info,
        gluten-free ingredient info, and potential sandwich objects.
        """
        super().__init__(task)

        # Identify children that need to be served
        self.goal_children = {get_parts(g)[1] for g in self.goals if get_parts(g) and get_parts(g)[0] == 'served'}

        # Extract allergy information
        self.allergic_children = {get_parts(s)[1] for s in self.static if match(s, "allergic_gluten", "*")}

        # Extract gluten-free ingredient information
        self.no_gluten_bread_types = {get_parts(s)[1] for s in self.static if match(s, "no_gluten_bread", "*")}
        self.no_gluten_content_types = {get_parts(s)[1] for s in self.static if match(s, "no_gluten_content", "*")}

        # Identify all potential sandwich objects from the initial state
        self.potential_sandwiches = set()
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] in ['notexist', 'at_kitchen_sandwich', 'ontray']:
                 if len(parts) > 1: # Ensure there's an object name
                    self.potential_sandwiches.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        state_facts = set(state) # Convert frozenset to set for faster lookups

        # 1. Identify waiting children and their needs
        waiting_children_at_table = {} # {child: table}
        # served_children = set() # Not strictly needed for heuristic calculation

        for child in self.goal_children:
            # Check if child is served
            if f'(served {child})' not in state_facts:
                # Check if child is waiting and where
                for fact in state_facts:
                    if match(fact, "waiting", child, "?table"):
                        table = get_parts(fact)[2]
                        waiting_children_at_table[child] = table
                        break # Found waiting location, move to next child

        # If all goal children are served, heuristic is 0
        if not waiting_children_at_table:
            return 0

        # Count needed sandwiches by type
        num_ng_needed = 0
        num_any_needed = 0
        for child in waiting_children_at_table:
            if child in self.allergic_children:
                num_ng_needed += 1
            else:
                num_any_needed += 1 # Non-allergic children can take any sandwich

        # 2. Count available ingredients in the kitchen
        bread_kitchen = set()
        content_kitchen = set()
        for fact in state_facts:
            if match(fact, "at_kitchen_bread", "?bread"):
                bread_kitchen.add(get_parts(fact)[1])
            elif match(fact, "at_kitchen_content", "?content"):
                content_kitchen.add(get_parts(fact)[1])

        # Count NG/G bread/content in kitchen
        num_ng_bread_k = len([b for b in bread_kitchen if b in self.no_gluten_bread_types])
        num_g_bread_k = len(bread_kitchen) - num_ng_bread_k
        num_ng_content_k = len([c for c in content_kitchen if c in self.no_gluten_content_types])
        num_g_content_k = len(content_kitchen) - num_ng_content_k

        # 3. Calculate number of makable sandwiches
        num_makable_ng = min(num_ng_bread_k, num_ng_content_k)
        # Assuming a regular sandwich can be made from any remaining bread/content
        total_makable = min(len(bread_kitchen), len(content_kitchen))
        num_makable_g = total_makable - num_makable_ng # Remaining makable are regular

        # 4. Categorize existing sandwiches by state and type
        sandwiches_state = {s: 'unknown' for s in self.potential_sandwiches}
        sandwiches_ng = set()
        tray_locations = {} # {tray: place}
        sandwiches_on_trays = {} # {sandw: tray}

        for fact in state_facts:
            if match(fact, "no_gluten_sandwich", "?sandw"):
                sandw = get_parts(fact)[1]
                sandwiches_ng.add(sandw)
            elif match(fact, "notexist", "?sandw"):
                 sandw = get_parts(fact)[1]
                 if sandw in self.potential_sandwiches:
                    sandwiches_state[sandw] = 'notexist'
            elif match(fact, "at_kitchen_sandwich", "?sandw"):
                 sandw = get_parts(fact)[1]
                 if sandw in self.potential_sandwiches:
                    sandwiches_state[sandw] = 'kitchen'
            elif match(fact, "ontray", "?sandw", "?tray"):
                 sandw, tray = get_parts(fact)[1:]
                 if sandw in self.potential_sandwiches:
                    sandwiches_on_trays[sandw] = tray
                    # Tentatively mark as on_tray, will refine with tray location
                    sandwiches_state[sandw] = 'on_tray'
            elif match(fact, "at", "?tray", "?place"):
                 tray, place = get_parts(fact)[1:]
                 tray_locations[tray] = place

        # Refine sandwich state based on tray location
        for sandw, tray in sandwiches_on_trays.items():
            if tray in tray_locations:
                if tray_locations[tray] == 'kitchen':
                    sandwiches_state[sandw] = 'on_tray_kitchen'
                # Assuming any other location is a table
                else: # tray_locations[tray] is a table
                    sandwiches_state[sandw] = 'at_table'
            else:
                 # Tray location unknown or tray not 'at' anywhere, treat as less ready (e.g., kitchen)
                 sandwiches_state[sandw] = 'on_tray_kitchen'


        # Count available sandwiches by type and state
        avail_ng_at_table = sum(1 for s in self.potential_sandwiches if sandwiches_state.get(s) == 'at_table' and s in sandwiches_ng)
        avail_g_at_table = sum(1 for s in self.potential_sandwiches if sandwiches_state.get(s) == 'at_table' and s not in sandwiches_ng)
        avail_ng_otk = sum(1 for s in self.potential_sandwiches if sandwiches_state.get(s) == 'on_tray_kitchen' and s in sandwiches_ng)
        avail_g_otk = sum(1 for s in self.potential_sandwiches if sandwiches_state.get(s) == 'on_tray_kitchen' and s not in sandwiches_ng)
        avail_ng_k = sum(1 for s in self.potential_sandwiches if sandwiches_state.get(s) == 'kitchen' and s in sandwiches_ng)
        avail_g_k = sum(1 for s in self.potential_sandwiches if sandwiches_state.get(s) == 'kitchen' and s not in sandwiches_ng)
        # For notexist, use makable counts
        avail_ng_ne = num_makable_ng
        avail_g_ne = num_makable_g

        # 5-8. Calculate cost using layered approach
        cost = 0

        # Cost 1: Serve (Sandwich is at table on a tray)
        use_ng_s = min(num_ng_needed, avail_ng_at_table)
        cost += use_ng_s * 1
        num_ng_needed -= use_ng_s
        avail_any_s = avail_g_at_table + (avail_ng_at_table - use_ng_s) # Remaining NG at table can serve Any
        use_any_s = min(num_any_needed, avail_any_s)
        cost += use_any_s * 1
        num_any_needed -= use_any_s

        # Cost 2: Move tray from kitchen + Serve (Sandwich on tray in kitchen)
        use_ng_otk = min(num_ng_needed, avail_ng_otk)
        cost += use_ng_otk * 2
        num_ng_needed -= use_ng_otk
        avail_any_otk = avail_g_otk + (avail_ng_otk - use_ng_otk) # Remaining NG on tray kitchen can serve Any
        use_any_otk = min(num_any_needed, avail_any_otk)
        cost += use_any_otk * 2
        num_any_needed -= use_any_otk

        # Cost 3: Put on tray + Move tray + Serve (Sandwich in kitchen)
        use_ng_k = min(num_ng_needed, avail_ng_k)
        cost += use_ng_k * 3
        num_ng_needed -= use_ng_k
        avail_any_k = avail_g_k + (avail_ng_k - use_ng_k) # Remaining NG in kitchen can serve Any
        use_any_k = min(num_any_needed, avail_any_k)
        cost += use_any_k * 3
        num_any_needed -= use_any_k

        # Cost 4: Make + Put on tray + Move tray + Serve (Sandwich notexist/makable)
        use_ng_ne = min(num_ng_needed, avail_ng_ne)
        cost += use_ng_ne * 4
        num_ng_needed -= use_ng_ne
        avail_any_ne = avail_g_ne + (avail_ng_ne - use_ng_ne) # Remaining makable NG can serve Any
        use_any_ne = min(num_any_needed, avail_any_ne)
        cost += use_any_ne * 4
        num_any_needed -= use_any_ne

        # If after exhausting all available/makable sandwiches, there are still
        # children needing service, it implies the problem is unsolvable with
        # the given resources (not enough bread/content/sandwiches).
        # The heuristic should ideally reflect this with a very high value,
        # but returning the calculated cost is sufficient for a non-admissible
        # heuristic in a greedy search, as it will still guide away from
        # states where needs cannot be met.
        # For simplicity and efficiency, we just return the calculated cost.
        # If num_ng_needed > 0 or num_any_needed > 0 here, the cost will be
        # based on the resources that *were* available.

        return cost
