from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string case defensively
    if not fact or not fact.strip():
        return []
    # Remove outer parentheses and split by spaces
    content = fact.strip()[1:-1]
    if not content: # Handle facts like "()" if they could occur
        return []
    return content.split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved
    children. It calculates a minimum estimated cost for each unserved child
    based on the current state of the "closest" available suitable sandwich
    and sums these individual costs.

    # Assumptions
    - Each child requires exactly one sandwich of a specific type (gluten-free
      if allergic, any type otherwise).
    - The sequence of actions to serve a child, starting from a non-existent
      sandwich, is Make -> Put on Tray -> Move Tray -> Serve (4 actions).
    - If a suitable sandwich exists in an intermediate state (e.g., in kitchen,
      on tray elsewhere), the initial steps (Make, Put on Tray) can be skipped,
      reducing the cost estimate for that child.
    - A tray is assumed to be available somewhere to facilitate Put on Tray and
      Move Tray actions when needed. Resource contention for trays or ingredients
      is not explicitly modeled in the cost calculation for individual children.
    - The problem is assumed to be solvable, meaning necessary ingredients and
      sandwich objects are implicitly available to make sandwiches if none exist.
    - The 'at' predicate in this domain is only used for trays.
    - 'waiting' facts are static for a given problem instance.

    # Heuristic Initialization
    - Extracts static information about which children are allergic and which
      bread/content portions are gluten-free.
    - Stores the goal conditions (which children need to be served).
    - Stores the static waiting places for each child.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Identify all children who are in the goal state but are not yet marked as `served` in the current state. These are the unserved children.
    2. For each unserved child:
       a. Determine the child's waiting location and allergy status (allergic or not).
       b. Determine the required sandwich type: gluten-free if the child is allergic, any type otherwise.
       c. Find the minimum estimated number of actions required to get a suitable sandwich to this child and serve them, based on the current state:
          - **Cost 1:** If a suitable sandwich is already on a tray at the child's location (`ontray S T` and `at T P`): 1 action (Serve).
          - **Cost 2:** If no such sandwich exists, but a suitable sandwich is on a tray elsewhere (`ontray S T` and `at T P'` where `P' != P`): 2 actions (Move Tray + Serve).
          - **Cost 3:** If no such sandwich exists on any tray, but a suitable sandwich is in the kitchen (`at_kitchen_sandwich S`): 3 actions (Put on Tray + Move Tray + Serve).
          - **Cost 4:** If no suitable sandwich exists in any of the above states (must be made): 4 actions (Make + Put on Tray + Move Tray + Serve).
       d. The minimum cost for this child is the lowest cost found among the applicable states (checked in order: 1, 2, 3, 4).
    3. The total heuristic value is the sum of the minimum estimated costs for all unserved children.
    4. If all children are served, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.

        @param task: The planning task object.
        """
        self.goals = task.goals  # Goal conditions, used to identify unserved children.
        static_facts = task.static # Facts that are true in all states.

        # Extract static information
        self.allergic_children = {get_parts(f)[1] for f in static_facts if match(f, "allergic_gluten", "*")}
        self.not_allergic_children = {get_parts(f)[1] for f in static_facts if match(f, "not_allergic_gluten", "*")}
        self.no_gluten_breads = {get_parts(f)[1] for f in static_facts if match(f, "no_gluten_bread", "*")}
        self.no_gluten_contents = {get_parts(f)[1] for f in static_facts if match(f, "no_gluten_content", "*")}

        # Extract static waiting places from initial state
        self.child_waiting_places = {get_parts(f)[1]: get_parts(f)[2] for f in task.initial_state if match(f, "waiting", "*", "*")}


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        @param node: The search node containing the current state.
        @return: The estimated cost to reach a goal state.
        """
        state = node.state  # Current world state as a frozenset of facts.

        # 1. Identify unserved children
        # Children in the goal list who are not marked as served in the current state.
        unserved_children = {
            get_parts(g)[1] for g in self.goals
            if match(g, "served", "*") and g not in state
        }

        # If all children are served, the heuristic is 0.
        if not unserved_children:
            return 0

        total_cost = 0

        # Pre-process state for quick lookups of sandwich/tray locations and types
        sandwiches_on_tray = {} # {sandwich: tray}
        tray_locations = {} # {tray: place}
        kitchen_sandwiches = set() # {sandwich}
        no_gluten_sandwiches = set() # {sandwich}

        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1:]
                sandwiches_on_tray[s] = t
            elif match(fact, "at", "*", "*"):
                 # In childsnacks domain, 'at' predicate is only used for trays.
                 t, p = get_parts(fact)[1:]
                 tray_locations[t] = p
            elif match(fact, "at_kitchen_sandwich", "*"):
                s = get_parts(fact)[1]
                kitchen_sandwiches.add(s)
            elif match(fact, "no_gluten_sandwich", "*"):
                s = get_parts(fact)[1]
                no_gluten_sandwiches.add(s)

        # 2. Calculate minimum cost for each unserved child
        for child in unserved_children:
            place = self.child_waiting_places.get(child)
            # Should always find the place if the problem is well-formed and child was initially waiting
            if place is None:
                 # This child was in the goal but not initially waiting? Or problem error.
                 # Treat as unsolvable for this child within this heuristic's scope?
                 # Or assign a high cost? Let's assume well-formed problems.
                 continue

            is_allergic = child in self.allergic_children

            # Default cost if sandwich needs to be made from scratch
            min_cost_for_child = 4 # Make + Put + Move + Serve

            # Check for existing suitable sandwiches in state, prioritizing closer ones

            # Check State 1: Suitable sandwich on tray at correct place (Cost 1)
            found_state1 = False
            for s, t in sandwiches_on_tray.items():
                if tray_locations.get(t) == place:
                    # Check if sandwich 's' is suitable for 'child'
                    s_is_gf = s in no_gluten_sandwiches
                    is_suitable = (is_allergic and s_is_gf) or (not is_allergic) # Non-allergic can eat any

                    if is_suitable:
                        min_cost_for_child = 1
                        found_state1 = True
                        break # Found the best case for this child

            if found_state1:
                total_cost += min_cost_for_child
                continue # Move to the next unserved child

            # Check State 2: Suitable sandwich on tray elsewhere (Cost 2)
            found_state2 = False
            for s, t in sandwiches_on_tray.items():
                 # Already checked tray_locations.get(t) == place in State 1
                 current_tray_place = tray_locations.get(t)
                 if current_tray_place is not None and current_tray_place != place:
                    # Check if sandwich 's' is suitable for 'child'
                    s_is_gf = s in no_gluten_sandwiches
                    is_suitable = (is_allergic and s_is_gf) or (not is_allergic)

                    if is_suitable:
                        min_cost_for_child = 2
                        found_state2 = True
                        break # Found the next best case for this child

            if found_state2:
                total_cost += min_cost_for_child
                continue # Move to the next unserved child

            # Check State 3: Suitable sandwich at kitchen (Cost 3)
            found_state3 = False
            for s in kitchen_sandwiches:
                # Check if sandwich 's' is suitable for 'child'
                s_is_gf = s in no_gluten_sandwiches
                is_suitable = (is_allergic and s_is_gf) or (not is_allergic)

                if is_suitable:
                    min_cost_for_child = 3
                    found_state3 = True
                    break # Found the next best case for this child

            if found_state3:
                total_cost += min_cost_for_child
                continue # Move to the next unserved child

            # If none of the above, the sandwich needs to be made (Cost 4)
            # min_cost_for_child is already initialized to 4.
            total_cost += min_cost_for_child

        return total_cost
