from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the number of unserved children and estimates the minimum cost to get a
    suitable sandwich to each child, prioritizing children who can be served with
    fewer actions based on the current state of available sandwiches and resources.

    # Assumptions
    - Any tray can be moved to any location in 1 action.
    - Enough trays are available for putting sandwiches on.
    - If bread/content/notexist sandwich objects are available, a sandwich can be made.
    - The problem is solvable (enough total resources exist in the initial state).
    - The cost of each action is 1.

    # Heuristic Initialization
    - Extract static facts about child allergies.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are currently 'waiting' but not 'served'.
    2. Determine the allergy status and waiting place for each unserved child.
    3. Count available sandwiches in different states:
       - On a tray at a specific place (mapping place to list of sandwiches).
       - In the kitchen.
    4. Determine which sandwiches are gluten-free.
    5. Count available bread and content portions in the kitchen, distinguishing gluten-free.
    6. Count available 'notexist' sandwich objects.
    7. Assign available sandwiches (from states 3) to unserved children greedily, prioritizing:
       - Allergic children needing gluten-free sandwiches.
       - Sandwiches that require fewer actions to reach the child:
         - Cost 1: Suitable sandwich on a tray already at the child's waiting place.
         - Cost 2: Suitable sandwich on a tray elsewhere (requires move + serve).
         - Cost 3: Suitable sandwich in the kitchen (requires put + move + serve).
    8. For any remaining unserved children, estimate the cost of making a suitable sandwich and delivering it (Cost 4: make + put + move + serve), consuming available bread, content, and 'notexist' resources greedily (prioritizing NG ingredients for allergic children, then using any available ingredients for non-allergic).
    9. Sum the costs for each child based on the assigned sandwich state or making cost.
    10. The total sum is the heuristic value. If all children are served, the heuristic is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts."""
        super().__init__(task)
        # Map child to their allergy status
        self.is_allergic = {
            get_parts(fact)[1]: True
            for fact in self.static
            if match(fact, "allergic_gluten", "*")
        }

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children and their places
        unserved = []
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:]
                # Check if the child is served in the current state
                if not any(match(f, "served", child) for f in state):
                     unserved.append((child, place, self.is_allergic.get(child, False)))

        # If all children are served, the heuristic is 0
        if not unserved:
            return 0

        # 3, 4. Count available sandwiches by state and type
        tray_locations = {} # map tray -> place
        for fact in state:
            if match(fact, "at", "*", "*") and fact[1:-1].split()[1].startswith('tray'):
                 tray, place = get_parts(fact)[1:]
                 tray_locations[tray] = place

        is_gluten_free_sandwich = {} # map sandwich -> bool
        # First identify explicitly NG sandwiches
        for fact in state:
            if match(fact, "no_gluten_sandwich", "*"):
                is_gluten_free_sandwich[get_parts(fact)[1]] = True
        # Assume non-gluten-free unless specified
        all_sandwiches_in_state = set()
        for fact in state:
             if match(fact, "at_kitchen_sandwich", "*"):
                 all_sandwiches_in_state.add(get_parts(fact)[1])
             elif match(fact, "ontray", "*", "*"):
                 all_sandwiches_in_state.add(get_parts(fact)[1])

        for s in all_sandwiches_in_state:
             if s not in is_gluten_free_sandwich:
                 is_gluten_free_sandwich[s] = False


        avail_s_ontray_at_place_ng = {} # map place -> list of s
        avail_s_ontray_at_place_reg = {} # map place -> list of s
        avail_s_kitchen_ng = [] # list of s
        avail_s_kitchen_reg = [] # list of s

        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1:]
                s_is_ng = is_gluten_free_sandwich.get(s, False)
                if t in tray_locations:
                    p = tray_locations[t]
                    if s_is_ng:
                        avail_s_ontray_at_place_ng.setdefault(p, []).append(s)
                    else:
                        avail_s_ontray_at_place_reg.setdefault(p, []).append(s)
            elif match(fact, "at_kitchen_sandwich", "*"):
                s = get_parts(fact)[1]
                s_is_ng = is_gluten_free_sandwich.get(s, False)
                if s_is_ng:
                    avail_s_kitchen_ng.append(s)
                else:
                    avail_s_kitchen_reg.append(s)

        # 5, 6. Count makeable resources
        count_ng_bread = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and any(match(sf, "no_gluten_bread", get_parts(fact)[1]) for sf in state))
        count_reg_bread = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and not any(match(sf, "no_gluten_bread", get_parts(fact)[1]) for sf in state))

        count_ng_content = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and any(match(sf, "no_gluten_content", get_parts(fact)[1]) for sf in state))
        count_reg_content = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and not any(match(sf, "no_gluten_content", get_parts(fact)[1]) for sf in state))

        avail_notexist = sum(1 for fact in state if match(fact, "notexist", "*"))
        avail_ng_bread = count_ng_bread
        avail_reg_bread = count_reg_bread
        avail_ng_content = count_ng_content
        avail_reg_content = count_reg_content


        # 7, 8. Greedy Assignment and Cost Calculation
        cost = 0
        used_sandwiches = set() # Track sandwiches by object name

        # Bucket 1 (Cost 1: Serve)
        # Allergic children first
        for c, p, is_allergic in list(unserved): # Iterate over a copy
            if is_allergic:
                if p in avail_s_ontray_at_place_ng and avail_s_ontray_at_place_ng[p]:
                    s = avail_s_ontray_at_place_ng[p].pop(0)
                    cost += 1
                    used_sandwiches.add(s)
                    unserved.remove((c, p, is_allergic))

        # Non-allergic children
        for c, p, is_allergic in list(unserved): # Iterate over a copy
            if not is_allergic:
                found_s = None
                if p in avail_s_ontray_at_place_reg and avail_s_ontray_at_place_reg[p]:
                     found_s = avail_s_ontray_at_place_reg[p].pop(0)
                elif p in avail_s_ontray_at_place_ng and avail_s_ontray_at_place_ng[p]:
                     found_s = avail_s_ontray_at_place_ng[p].pop(0)

                if found_s:
                    cost += 1
                    used_sandwiches.add(found_s)
                    unserved.remove((c, p, is_allergic))

        # Bucket 2 (Cost 2: Move + Serve)
        # Collect remaining ontray sandwiches not used in Bucket 1
        rem_ontray_ng = []
        rem_ontray_reg = []
        for p, s_list in avail_s_ontray_at_place_ng.items():
             rem_ontray_ng.extend([s for s in s_list if s not in used_sandwiches])
        for p, s_list in avail_s_ontray_at_place_reg.items():
             rem_ontray_reg.extend([s for s in s_list if s not in used_sandwiches])


        # Allergic children first
        for c, p, is_allergic in list(unserved): # Iterate over a copy
            if is_allergic:
                if rem_ontray_ng:
                    s = rem_ontray_ng.pop(0)
                    cost += 2
                    used_sandwiches.add(s)
                    unserved.remove((c, p, is_allergic))

        # Non-allergic children
        for c, p, is_allergic in list(unserved): # Iterate over a copy
            if not is_allergic:
                found_s = None
                if rem_ontray_reg:
                    found_s = rem_ontray_reg.pop(0)
                elif rem_ontray_ng:
                    found_s = rem_ontray_ng.pop(0)

                if found_s:
                    cost += 2
                    used_sandwiches.add(found_s)
                    unserved.remove((c, p, is_allergic))

        # Bucket 3 (Cost 3: Put + Move + Serve)
        # Sandwiches in kitchen not used
        rem_kitchen_ng = [s for s in avail_s_kitchen_ng if s not in used_sandwiches]
        rem_kitchen_reg = [s for s in avail_s_kitchen_reg if s not in used_sandwiches]

        # Allergic children first
        for c, p, is_allergic in list(unserved): # Iterate over a copy
            if is_allergic:
                if rem_kitchen_ng:
                    s = rem_kitchen_ng.pop(0)
                    cost += 3
                    used_sandwiches.add(s)
                    unserved.remove((c, p, is_allergic))

        # Non-allergic children
        for c, p, is_allergic in list(unserved): # Iterate over a copy
            if not is_allergic:
                found_s = None
                if rem_kitchen_reg:
                    found_s = rem_kitchen_reg.pop(0)
                elif rem_kitchen_ng:
                    found_s = rem_kitchen_ng.pop(0)

                if found_s:
                    cost += 3
                    used_sandwiches.add(found_s)
                    unserved.remove((c, p, is_allergic))

        # Bucket 4 (Cost 4: Make + Put + Move + Serve)
        # Sandwiches need to be made

        # Allergic children first
        for c, p, is_allergic in list(unserved): # Iterate over a copy
            if is_allergic:
                if avail_notexist > 0 and avail_ng_bread > 0 and avail_ng_content > 0:
                    cost += 4
                    avail_notexist -= 1
                    avail_ng_bread -= 1
                    avail_ng_content -= 1
                    unserved.remove((c, p, is_allergic))

        # Non-allergic children
        for c, p, is_allergic in list(unserved): # Iterate over a copy
            if not is_allergic:
                # Can use REG ingredients first, then NG ingredients if available
                made = False
                if avail_notexist > 0:
                    if avail_reg_bread > 0 and avail_reg_content > 0:
                        cost += 4
                        avail_notexist -= 1
                        avail_reg_bread -= 1
                        avail_reg_content -= 1
                        made = True
                    elif avail_ng_bread > 0 and avail_ng_content > 0: # Use NG if REG exhausted
                         cost += 4
                         avail_notexist -= 1
                         avail_ng_bread -= 1
                         avail_ng_content -= 1
                         made = True
                if made:
                    unserved.remove((c, p, is_allergic))

        # Any remaining children in 'unserved' cannot be served with available resources
        # based on this heuristic's counting. This might indicate unsolvability or
        # heuristic limitation. For a non-admissible heuristic, we just return the cost calculated so far.
        # We could add a large penalty here, but let's stick to the calculated cost.

        return cost
