from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children.
    It counts the necessary 'make sandwich', 'put on tray', 'move tray', and 'serve' actions
    by considering the current state of sandwiches, trays, and children's needs and locations.
    It attempts to account for resource sharing (e.g., trays serving children at the same location)
    in a relaxed manner.

    # Assumptions
    - Ingredients and potential sandwiches are sufficient to make any required sandwich if
      the necessary types of ingredients and a 'notexist' sandwich are present in the state.
    - Trays have sufficient capacity to hold needed sandwiches for a location (heuristic counts 1 move per location needing a tray).
    - Any tray can be moved to any location.
    - The kitchen is a place.

    # Heuristic Initialization
    - Identify all children who need to be served based on the goal state.
    - Store static facts like allergy status for children.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all children who are not yet served based on the goal state and current state. For each unserved child, determine their allergy status (needs GF or any sandwich) and their current waiting place.
    2. Count the total number of unserved children (`N_unserved`). This contributes `N_unserved` to the heuristic (for the 'serve' actions). If `N_unserved` is 0, the goal is reached, return 0.
    3. Count the number of suitable sandwiches already made (either `at_kitchen_sandwich` or `ontray`) in the current state, distinguishing between GF and regular suitable sandwiches.
    4. Calculate the number of GF and regular sandwiches that still need to be made to satisfy all unserved children's needs, after accounting for existing made sandwiches. This contributes to the heuristic (for 'make_sandwich' actions).
    5. Count the number of suitable sandwiches that are currently `ontray` in the current state, distinguishing between GF and regular. Greedily assign these to unserved children's needs (GF for GF first, then remaining GF for Reg, then Reg for Reg) to determine how many children's sandwich+ontray needs are met by existing on-tray resources.
    6. The number of sandwiches that need to be put on a tray is the total number of unserved children minus the number of children whose sandwich+ontray need is met by existing on-tray sandwiches. This contributes to the heuristic (for 'put_on_tray' actions).
    7. Identify the set of unique places where unserved children are waiting.
    8. Identify the set of unique places where trays are currently located in the current state.
    9. Count the number of places from step 7 that are *not* in the set from step 8. For each such place, a tray needs to be moved there. This contributes to the heuristic (for 'move_tray' actions).
    10. Sum up the costs from steps 2, 4, 6, and 9 to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        super().__init__(task)
        # Store goal children
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Store static facts for quick lookup
        self.child_allergy = {} # child -> True if allergic, False otherwise

        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == "allergic_gluten":
                self.child_allergy[parts[1]] = True
            elif predicate == "not_allergic_gluten":
                self.child_allergy[parts[1]] = False
            # no_gluten_bread/content/sandwich are not treated as static here, checked in state

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state
        total_cost = 0

        # First pass: Collect state information and identify unserved children
        unserved_children_info = {} # child -> {'place': p, 'needs_gf': bool}
        children_waiting_places = set() # Set of places where children are waiting

        available_gf_sandwiches_kitchen = 0
        available_reg_sandwiches_kitchen = 0
        available_gf_sandwiches_ontray = 0
        available_reg_sandwiches_ontray = 0

        trays_at_place = {} # place -> count

        sandwich_is_gf = {} # Map sandwich name to True if GF, False otherwise (based on state)
        sandwich_on_tray = set() # Set of sandwich names on a tray
        sandwich_in_kitchen = set() # Set of sandwich names in kitchen
        tray_location = {} # Map tray name to place
        all_children_in_problem = set() # Collect all children mentioned in waiting facts

        served_children = set() # Collect children already served

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == "waiting":
                child, place = parts[1], parts[2]
                all_children_in_problem.add(child) # Collect all children mentioned

            elif predicate == "served":
                 child = parts[1]
                 served_children.add(child)

            elif predicate == "at_kitchen_sandwich":
                s = parts[1]
                sandwich_in_kitchen.add(s)
            elif predicate == "ontray":
                s, t = parts[1], parts[2]
                sandwich_on_tray.add(s)
            elif predicate == "no_gluten_sandwich":
                 s = parts[1]
                 sandwich_is_gf[s] = True
            # no_gluten_bread/content, at_kitchen_bread/content, notexist are not needed for this simplified heuristic
            elif predicate == "at": # Tray location
                 t, p = parts[1], parts[2]
                 tray_location[t] = p
                 trays_at_place[p] = trays_at_place.get(p, 0) + 1

        # Now identify unserved children from the set of all children in the problem
        # who are also in the goal state and not yet served.
        for child in all_children_in_problem:
             served_fact = "(served {})".format(child)
             # Check if this child is a goal child AND is not yet served
             if child in self.goal_children and child not in served_children:
                  # Find their waiting place in the current state
                  waiting_place = None
                  for fact in state: # Need to iterate state again to find waiting place
                       if match(fact, "waiting", child, "*"):
                            waiting_place = get_parts(fact)[2]
                            break
                  if waiting_place: # Should always find one if child is waiting and not served
                       unserved_children_info[child] = {'place': waiting_place, 'needs_gf': self.child_allergy.get(child, False)}
                       children_waiting_places.add(waiting_place)
                  # else: child is a goal and waiting fact is missing? Assume valid states.


        # Count available made sandwiches by type and location
        for s in sandwich_in_kitchen:
             is_gf = sandwich_is_gf.get(s, False)
             if is_gf:
                 available_gf_sandwiches_kitchen += 1
             else:
                 available_reg_sandwiches_kitchen += 1

        for s in sandwich_on_tray:
             is_gf = sandwich_is_gf.get(s, False)
             if is_gf:
                 available_gf_sandwiches_ontray += 1
             else:
                 available_reg_sandwiches_ontray += 1

        available_gf_sandwiches_total = available_gf_sandwiches_kitchen + available_gf_sandwiches_ontray
        available_reg_sandwiches_total = available_reg_sandwiches_kitchen + available_reg_sandwiches_ontray


        # 2. Calculate costs based on unserved children's needs

        num_unserved = len(unserved_children_info)
        if num_unserved == 0:
            return 0 # Goal reached

        # Cost Component 1: Serve actions
        total_cost += num_unserved

        # Count needed sandwiches by type
        needed_gf = sum(1 for info in unserved_children_info.values() if info['needs_gf'])
        needed_reg = sum(1 for info in unserved_children_info.values() if not info['needs_gf']) # Non-allergic can take Reg or GF

        # Sandwiches to make
        # Prioritize using existing made sandwiches (kitchen or ontray)
        # Greedily assign existing made sandwiches to needed counts
        use_gf_made_for_gf = min(needed_gf, available_gf_sandwiches_total)
        needed_gf -= use_gf_made_for_gf
        available_gf_sandwiches_total -= use_gf_made_for_gf # Update remaining available GF

        use_gf_made_for_reg = min(needed_reg, available_gf_sandwiches_total)
        needed_reg -= use_gf_made_for_reg
        # available_gf_sandwiches_total -= use_gf_made_for_reg # No need to track remaining available made after assignment

        use_reg_made_for_reg = min(needed_reg, available_reg_sandwiches_total)
        needed_reg -= use_reg_made_for_reg
        # available_reg_sandwiches_total -= use_reg_made_for_reg # No need to track remaining available made after assignment

        gf_to_make = needed_gf
        reg_to_make = needed_reg

        # Cost Component 2: Make actions
        total_cost += gf_to_make + reg_to_make

        # Cost Component 3: Put on Tray actions
        # Count how many unserved children can be served by sandwiches *already on trays*.
        # This requires matching needs (GF/Reg) with available on-tray sandwiches (GF/Reg).
        # Greedily assign existing on-tray sandwiches to needed children counts
        needed_gf_initial = sum(1 for info in unserved_children_info.values() if info['needs_gf'])
        needed_reg_initial = sum(1 for info in unserved_children_info.values() if not info['needs_gf'])

        served_by_ontray_gf = min(needed_gf_initial, available_gf_sandwiches_ontray)
        remaining_gf_needed = needed_gf_initial - served_by_ontray_gf
        remaining_reg_needed = needed_reg_initial

        available_gf_sandwiches_ontray_temp = available_gf_sandwiches_ontray - served_by_ontray_gf # Use a temp count

        served_by_ontray_gf_for_reg = min(remaining_reg_needed, available_gf_sandwiches_ontray_temp)
        remaining_reg_needed -= served_by_ontray_gf_for_reg

        served_by_ontray_reg = min(remaining_reg_needed, available_reg_sandwiches_ontray)
        # remaining_reg_needed -= served_by_ontray_reg # Not needed

        num_served_by_existing_ontray = served_by_ontray_gf + served_by_ontray_gf_for_reg + served_by_ontray_reg

        # The number of sandwiches that need to be put on a tray is the total needed minus those already on trays.
        num_put_on_tray = num_unserved - num_served_by_existing_ontray
        num_put_on_tray = max(0, num_put_on_tray) # Ensure non-negative
        total_cost += num_put_on_tray


        # Cost Component 4: Move Tray actions
        # We need a tray at each unique place where unserved children are waiting, UNLESS a tray is already there.
        needed_places = children_waiting_places
        trays_already_at_needed_places = set()
        for place in needed_places:
             if trays_at_place.get(place, 0) > 0:
                 trays_already_at_needed_places.add(place)

        places_needing_tray_moved = needed_places - trays_already_at_needed_places
        total_cost += len(places_needing_tray_moved)

        return total_cost

