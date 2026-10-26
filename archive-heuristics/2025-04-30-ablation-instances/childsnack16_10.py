from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all unserved children.
    It calculates the cost for each unserved child independently, based on the
    current location and status of suitable sandwiches, and sums these individual costs.

    # Assumptions
    - Each unserved child requires one suitable sandwich (gluten-free for allergic children, any for others).
    - The sandwich must be on a tray and the tray must be at the child's waiting location for the 'serve' action.
    - Tray capacity is sufficient (a tray can hold a needed sandwich).
    - Ingredient and 'notexist' sandwich slot availability for making new sandwiches is sufficient whenever needed.
    - All actions have an implicit cost of 1.
    - Resource sharing (like trays or kitchen access) is not explicitly modeled in the cost calculation; costs are summed per child as if resources were dedicated.

    # Heuristic Initialization
    The heuristic extracts static information from the task definition:
    - The waiting location for each child.
    - The allergy status (allergic or not) for each child.
    - The types of bread and content that are gluten-free.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:
    1. Initialize the total heuristic cost to 0.
    2. Identify all children who have not yet been served by checking the '(served ?c)' facts in the current state.
    3. For each unserved child:
       a. Add 1 to the total cost, representing the final 'serve' action required for this child.
       b. Determine the child's waiting location and allergy status using the static information.
       c. Check if a suitable sandwich (gluten-free if the child is allergic, any otherwise) is currently available on a tray located at the child's waiting place.
          - If such a sandwich exists and is correctly positioned, this child is considered 'ready to serve' in terms of sandwich delivery. No further cost is added for this child's delivery step.
       d. If the child is not 'ready to serve' at their location, determine the minimum estimated actions required to get a suitable sandwich onto a tray and deliver it to the child's location:
          i. Check if a suitable sandwich is available on a tray located *anywhere else* (not at the child's place).
             - If yes, it needs 1 action ('move_tray') to bring the tray to the child's location. Add 1 to the total cost.
          ii. If no suitable sandwich is on any tray, check if a suitable sandwich is available *in the kitchen* (predicate 'at_kitchen_sandwich').
              - If yes, it needs 1 action ('put_on_tray') to put it on a tray (assuming a tray is available in the kitchen) and 1 action ('move_tray') to bring the tray to the child's location. Add 2 to the total cost.
          iii. If no suitable sandwich is available on any tray or in the kitchen, assume a new suitable sandwich must be made. (This step assumes necessary ingredients and a 'notexist' sandwich slot are available).
               - It needs 1 action ('make_sandwich' or 'make_sandwich_no_gluten'), 1 action ('put_on_tray'), and 1 action ('move_tray'). Add 3 to the total cost.
    4. The final total cost is the heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Child waiting locations.
        - Child allergy status.
        - Gluten-free ingredient types.
        """
        self.goals = task.goals # Not strictly needed for this heuristic logic, but available
        static_facts = task.static

        self.child_place = {}
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.all_children = set()
        self.no_gluten_bread_types = set()
        self.no_gluten_content_types = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "waiting" and len(parts) == 3:
                child, place = parts[1:]
                self.child_place[child] = place
                self.all_children.add(child)
            elif predicate == "allergic_gluten" and len(parts) == 2:
                child = parts[1]
                self.allergic_children.add(child)
            elif predicate == "not_allergic_gluten" and len(parts) == 2:
                child = parts[1]
                self.not_allergic_children.add(child)
            elif predicate == "no_gluten_bread" and len(parts) == 2:
                bread = parts[1]
                self.no_gluten_bread_types.add(bread)
            elif predicate == "no_gluten_content" and len(parts) == 2:
                content = parts[1]
                self.no_gluten_content_types.add(content)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse current state facts ---
        served_children = set()
        gf_sandwiches_in_state = set()
        sandwich_locs = {} # map sandwich -> 'kitchen' or 'ontray_t'
        tray_locs = {} # map tray -> place
        ontray_map = {} # map tray -> set of sandwiches
        kitchen_sandwiches = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == "served" and len(parts) == 2:
                served_children.add(parts[1])
            elif predicate == "no_gluten_sandwich" and len(parts) == 2:
                gf_sandwiches_in_state.add(parts[1])
            elif predicate == "at" and len(parts) == 3:
                 # Assuming (at ?t ?p) facts in state always refer to trays based on domain
                 tray, place = parts[1:]
                 tray_locs[tray] = place
            elif predicate == "at_kitchen_sandwich" and len(parts) == 2:
                sandwich = parts[1]
                sandwich_locs[sandwich] = 'kitchen'
                kitchen_sandwiches.add(sandwich)
            elif predicate == "ontray" and len(parts) == 3:
                sandwich, tray = parts[1:]
                sandwich_locs[sandwich] = f'ontray_{tray}'
                ontray_map.setdefault(tray, set()).add(sandwich)


        # --- Compute heuristic cost ---
        total_cost = 0

        for child in self.all_children:
            # If child is already served, no cost needed for this child
            if child in served_children:
                continue

            # Child needs to be served
            total_cost += 1 # Cost for the 'serve' action

            child_place = self.child_place.get(child) # Get place from static info
            if child_place is None:
                 # This child is waiting according to static info but has no place?
                 # Should not happen in valid problems. Skip or add large cost.
                 # Skipping assumes valid problem definition.
                 continue

            is_allergic = child in self.allergic_children

            # Check if a suitable sandwich is already on a tray at the child's place
            suitable_sandwich_ready_at_place = False
            for tray, place in tray_locs.items():
                if place == child_place:
                    if tray in ontray_map:
                        for s in ontray_map[tray]:
                            is_s_gf = s in gf_sandwiches_in_state
                            # Check suitability: GF for allergic, Any for non-allergic
                            if (is_allergic and is_s_gf) or (not is_allergic):
                                suitable_sandwich_ready_at_place = True
                                break # Found one suitable sandwich on a tray at the correct place
                    if suitable_sandwich_ready_at_place:
                        break # Found a tray at the correct place with a suitable sandwich

            # If child is not ready to be served from their location, add cost to get sandwich/tray there
            if not suitable_sandwich_ready_at_place:
                # Find the "closest" source for a suitable sandwich
                best_cost_to_get_ready = float('inf')

                # Option 1: Suitable sandwich on a tray elsewhere
                found_ontray_elsewhere = False
                for s, s_loc in sandwich_locs.items():
                    if s_loc.startswith('ontray_'):
                        tray = s_loc.split('_')[1]
                        # Check if the tray is NOT at the child's place
                        if tray in tray_locs and tray_locs[tray] != child_place:
                            is_s_gf = s in gf_sandwiches_in_state
                            if (is_allergic and is_s_gf) or (not is_allergic):
                                # Cost to get here: 1 (move_tray)
                                best_cost_to_get_ready = min(best_cost_to_get_ready, 1)
                                found_ontray_elsewhere = True
                                # Don't break, continue searching for the minimum cost source

                # Option 2: Suitable sandwich in the kitchen
                found_kitchen_sandwich = False
                for s in kitchen_sandwiches:
                    is_s_gf = s in gf_sandwiches_in_state
                    if (is_allergic and is_s_gf) or (not is_allergic):
                        # Cost to get here: 1 (put_on_tray) + 1 (move_tray)
                        best_cost_to_get_ready = min(best_cost_to_get_ready, 1 + 1)
                        found_kitchen_sandwich = True
                        # Don't break

                # Option 3: Need to make a suitable sandwich
                # Assume making is always possible if needed (ignoring ingredient/slot limits for simplicity)
                # This option is only considered if no suitable sandwich exists anywhere else.
                if not found_ontray_elsewhere and not found_kitchen_sandwich:
                     # Cost to get here: 1 (make) + 1 (put_on_tray) + 1 (move_tray)
                     best_cost_to_get_ready = min(best_cost_to_get_ready, 1 + 1 + 1)


                # Add the minimum cost to get the sandwich/tray ready for this child
                # If best_cost_to_get_ready is still infinity, it implies no way to get a sandwich
                # under the heuristic's assumptions (e.g., no existing sandwiches, and making is not possible
                # if we weren't assuming it's always possible).
                # For this heuristic, we assume making is always possible if needed, so best_cost_to_get_ready
                # will be at most 3 if not already 0, 1, or 2.
                if best_cost_to_get_ready != float('inf'):
                     total_cost += best_cost_to_get_ready
                # else: # Should not happen in solvable problems under heuristic assumptions
                #     pass # Or return float('inf') if unsolvable states are possible

        return total_cost
