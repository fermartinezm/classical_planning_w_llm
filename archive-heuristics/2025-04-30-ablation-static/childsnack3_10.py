# Assume Heuristic base class is available from 'heuristics.heuristic_base'
# from heuristics.heuristic_base import Heuristic

# If running this code requires a base class definition and heuristic_base is not available,
# uncomment the following dummy class definition:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts in the fact is at least the number of arguments in the pattern
    if len(parts) < len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the required number of actions to reach a goal state
    by summing the minimum estimated costs for each unserved child independently.
    It relaxes resource contention (e.g., multiple children needing the same sandwich
    or limited ingredients/trays) to provide an efficiently computable estimate.

    # Assumptions
    - Actions have a unit cost of 1.
    - Resources (sandwiches, trays, ingredients) are considered available for each
      child's calculation without strict depletion (relaxation).
    - Sufficient tray capacity and availability are assumed for 'put_on_tray'
      and 'move_tray' actions.
    - Sufficient available sandwich names (`notexist`) are assumed if at least
      one `notexist` fact exists.
    - Sufficient bread and content portions are assumed if at least one of each
      required type (GF/Any) exists in the kitchen.
    - Base costs for serving a child, depending on the state of a suitable sandwich:
        - Sandwich is on a tray at the child's location: 1 action (serve).
        - Sandwich is on a tray at a different location: 2 actions (move tray, serve).
        - Sandwich is in the kitchen: 3 actions (put on tray, move tray, serve).
        - Sandwich needs to be made: 4 actions (make, put on tray, move tray, serve).

    # Heuristic Initialization
    - Extracts and stores the set of goal facts (which specify which children need to be served).
    - Extracts and stores the allergy status (allergic_gluten or not_allergic_gluten)
      for all children mentioned in the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unserved Children: Determine which children from the goal state
       are not currently marked as `(served ?c)` in the current state. For each
       unserved child, find their waiting place `?p` from the `(waiting ?c ?p)` fact
       and retrieve their allergy status (allergic or not) stored during initialization.

    2. Collect State Information: Scan the current state facts to gather information about:
       - The location of each tray (`(at ?t ?p)`).
       - The location of each existing sandwich (`(ontray ?s ?t)` or `(at_kitchen_sandwich ?s)`).
       - Which sandwiches are gluten-free (`(no_gluten_sandwich ?s)`).
       - Which sandwich names are available to be made (`(notexist ?s)`).
       - The availability of bread and content portions in the kitchen, distinguishing
         between gluten-free and any type (`(at_kitchen_bread ?b)`, `(no_gluten_bread ?b)`,
         `(at_kitchen_content ?c)`, `(no_gluten_content ?c)`).

    3. Determine Resource Availability for New Sandwiches: Check if it's possible
       to make a new gluten-free sandwich (requires at least one GF bread, one GF content,
       and one `notexist` sandwich name) and if it's possible to make a new any
       sandwich (requires at least one any bread, one any content, and one `notexist`
       sandwich name).

    4. Calculate Minimum Cost Per Unserved Child: For each unserved child, estimate
       the minimum number of actions required to serve them. This minimum is found
       by considering all suitable sandwiches (gluten-free for allergic children,
       any sandwich for non-allergic children) and the option to make a new suitable
       sandwich.
       - For an existing suitable sandwich `S`:
         - If `S` is on a tray `T` at the child's waiting place `P`: Cost is 1 (serve).
         - If `S` is on a tray `T` at a different place `P'`: Cost is 2 (move tray, serve).
         - If `S` is in the kitchen: Cost is 3 (put on tray, move tray, serve).
       - For making a new suitable sandwich:
         - If the necessary ingredients and an available sandwich name exist: Cost is 4 (make, put on tray, move tray, serve).
         - Otherwise, this option has infinite cost.
       The minimum cost for the child is the smallest of these calculated costs.

    5. Sum Costs: The total heuristic value is the sum of the minimum costs calculated
       for each unserved child. If the minimum cost for any child is infinite (meaning
       they cannot be served from the current state), the total heuristic is infinite.

    6. Goal State: If the set of unserved children is empty, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals  # Goal conditions, e.g., {(served child1), (served child2)}

        # Extract allergy status for all children from static facts
        self.is_allergic = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.is_allergic[parts[1]] = True
            elif parts[0] == 'not_allergic_gluten':
                self.is_allergic[parts[1]] = False

        # Identify all children who need to be served from the goals
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to serve
        all unserved children.
        """
        state = node.state  # Current world state (frozenset of facts)

        # 1. Identify Unserved Children and their locations/allergy status
        unserved_children_info = [] # List of (child_name, place, is_allergic)
        waiting_locations = {} # {child_name: place}
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                _, child, place = get_parts(fact)
                waiting_locations[child] = place

        for child in self.goal_children:
            if "(served {})".format(child) not in state:
                # Child is unserved
                place = waiting_locations.get(child) # Get waiting place
                is_allergic = self.is_allergic.get(child, False) # Default to not allergic if status missing
                if place is not None: # Only consider children who are waiting somewhere
                     unserved_children_info.append((child, place, is_allergic))
                # Note: If a child is in goal but not waiting, they are likely unreachable/problematic,
                # but we assume valid problems where goal children are initially waiting.

        # If all goal children are served, heuristic is 0
        if not unserved_children_info:
            return 0

        # 2. Collect State Information about sandwiches, trays, and resources
        sandwiches_ontray = {} # {sandwich_name: tray_name}
        sandwiches_kitchen = set() # {sandwich_name}
        sandwich_names_notexist = set() # {sandwich_name}
        gf_sandwiches = set() # {sandwich_name}
        tray_locations = {} # {tray_name: place_name}

        num_gf_bread = 0
        num_any_bread = 0
        num_gf_content = 0
        num_any_content = 0

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'ontray':
                s, t = parts[1], parts[2]
                sandwiches_ontray[s] = t
            elif parts[0] == 'at_kitchen_sandwich':
                s = parts[1]
                sandwiches_kitchen.add(s)
            elif parts[0] == 'notexist':
                s = parts[1]
                sandwich_names_notexist.add(s)
            elif parts[0] == 'no_gluten_sandwich':
                s = parts[1]
                gf_sandwiches.add(s)
            elif parts[0] == 'at':
                 # Check if it's a tray location fact
                 if len(parts) == 3 and parts[1].startswith('tray'): # Simple check for tray type
                     t, p = parts[1], parts[2]
                     tray_locations[t] = p
            elif parts[0] == 'at_kitchen_bread':
                num_any_bread += 1
                b = parts[1]
                # Check if the corresponding no_gluten_bread fact exists for this specific bread instance
                if "(no_gluten_bread {})".format(b) in state:
                    num_gf_bread += 1
            elif parts[0] == 'at_kitchen_content':
                num_any_content += 1
                c = parts[1]
                 # Check if the corresponding no_gluten_content fact exists for this specific content instance
                if "(no_gluten_content {})".format(c) in state:
                    num_gf_content += 1

        # All existing sandwiches are those currently available (on tray or in kitchen)
        existing_sandwiches = sandwiches_ontray.keys() | sandwiches_kitchen

        # 3. Determine Resource Availability for New Sandwiches
        # Simplified check: can make if at least one of each resource type exists and a name is available
        can_make_GF = num_gf_bread > 0 and num_gf_content > 0 and len(sandwich_names_notexist) > 0
        can_make_Any = num_any_bread > 0 and num_any_content > 0 and len(sandwich_names_notexist) > 0


        # 4. Calculate Minimum Cost Per Unserved Child
        total_h = 0
        for child, child_place, is_allergic in unserved_children_info:
            min_cost_C = float('inf')

            # Determine suitable sandwiches based on allergy status
            # Suitable existing sandwiches are GF if allergic, Any if not.
            suitable_existing_sandwiches = existing_sandwiches.intersection(gf_sandwiches) if is_allergic else existing_sandwiches

            # Check existing suitable sandwiches
            for s in suitable_existing_sandwiches:
                if s in sandwiches_ontray:
                    t = sandwiches_ontray[s]
                    if t in tray_locations:
                        tray_place = tray_locations[t]
                        if tray_place == child_place:
                            min_cost_C = min(min_cost_C, 1) # Serve
                        else:
                            min_cost_C = min(min_cost_C, 2) # Move + Serve
                elif s in sandwiches_kitchen:
                    min_cost_C = min(min_cost_C, 3) # Put + Move + Serve

            # Check possibility of making a new suitable sandwich
            cost_make_new = float('inf')
            if len(sandwich_names_notexist) > 0: # Need an available name
                if is_allergic:
                    if can_make_GF:
                        cost_make_new = 4 # Make GF + Put + Move + Serve
                else: # not allergic
                    if can_make_Any:
                        cost_make_new = 4 # Make Any + Put + Move + Serve

            min_cost_C = min(min_cost_C, cost_make_new)

            # If a child cannot be served even with making a new sandwich, the state is likely unsolvable
            if min_cost_C == float('inf'):
                return float('inf')

            total_h += min_cost_C

        # 5. Sum Costs (already done in the loop)
        return total_h
