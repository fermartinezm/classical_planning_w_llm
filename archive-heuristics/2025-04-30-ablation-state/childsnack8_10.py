from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
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
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of unserved children and adds costs for the necessary
    steps (making sandwiches, putting them on trays, moving trays) required
    to get suitable sandwiches onto trays and those trays to the children's locations.
    It assumes a pipeline flow for sandwiches: Make -> PutOnTray -> MoveTray -> Serve.

    # Assumptions
    - Each child requires one suitable sandwich (GF for allergic, any for non-allergic).
    - Trays have unlimited capacity for sandwiches.
    - Ingredients (bread, content) and 'notexist' slots are sufficient to make needed sandwiches.
    - Trays are available somewhere to be moved to needed locations.
    - Action costs are uniform (cost 1).
    - All unserved children who need serving are in a 'waiting' state.

    # Heuristic Initialization
    - Identify all children and their allergy status from static facts.
    - Identify all potential objects (children, sandwiches, trays, places) from initial state and static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all unserved children, their waiting places, and allergy status.
    2. If no children are unserved, the heuristic is 0 (goal state).
    3. Count the total number of unserved children (N_unserved), and separate into allergic (N_allergic) and non-allergic (N_not_allergic). This contributes N_unserved to the heuristic (for the final 'serve' action for each).
    4. Count the number of gluten-free and regular sandwiches currently in the kitchen and on trays.
    5. Calculate the number of gluten-free and regular sandwiches that still need to be made to satisfy the demand of unserved children (total needed minus those already existing). Add this count to the heuristic (cost for 'make' actions).
    6. Calculate the number of gluten-free and regular sandwiches that still need to be put on trays (i.e., are not already on trays) to satisfy the demand. Add this count to the heuristic (cost for 'put_on_tray' actions).
    7. Identify all places where unserved children are waiting.
    8. Identify all places where trays are currently located.
    9. Count the number of places where children are waiting but no tray is present. Each such place requires a 'move_tray' action. Add this count to the heuristic.
    10. The total heuristic is the sum of costs from steps 3, 5, 6, and 9.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and objects.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Extract child allergy status from static facts
        self.child_allergy = {}
        for fact in self.static_facts:
            if match(fact, "allergic_gluten", "*"):
                child_name = get_parts(fact)[1]
                self.child_allergy[child_name] = True
            elif match(fact, "not_allergic_gluten", "*"):
                child_name = get_parts(fact)[1]
                self.child_allergy[child_name] = False

        # Extract all potential objects from initial state and static facts
        all_facts_set = set(self.initial_state) | set(self.static_facts)

        self.all_children = set()
        self.all_sandwiches = set()
        self.all_trays = set()
        self.all_places = set()

        for fact in all_facts_set:
             parts = get_parts(fact)
             if not parts: continue

             predicate = parts[0]
             if predicate in ["served", "waiting", "allergic_gluten", "not_allergic_gluten"]:
                 if len(parts) > 1: self.all_children.add(parts[1])
             elif predicate in ["at_kitchen_bread", "at_kitchen_content", "no_gluten_bread", "no_gluten_content"]:
                 pass # Ingredients
             elif predicate in ["at_kitchen_sandwich", "ontray", "no_gluten_sandwich", "notexist"]:
                 if len(parts) > 1: self.all_sandwiches.add(parts[1])
             elif predicate == "ontray":
                 if len(parts) > 2: # (ontray ?s ?t)
                     self.all_sandwiches.add(parts[1])
                     self.all_trays.add(parts[2])
             elif predicate == "at":
                 if len(parts) > 2: # (at ?t ?p)
                     self.all_trays.add(parts[1])
                     self.all_places.add(parts[2])
             elif predicate == "waiting":
                 if len(parts) > 2: # (waiting ?c ?p)
                     self.all_children.add(parts[1])
                     self.all_places.add(parts[2])

        # Add the constant kitchen place if not already found
        self.all_places.add("kitchen")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children and their waiting places and allergy status.
        unserved_children = {}
        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        for child in self.all_children: # Iterate through all known children
            if child not in served_children_in_state:
                # Find where the child is waiting
                waiting_place = None
                for fact in state:
                    if match(fact, "waiting", child, "*"):
                        waiting_place = get_parts(fact)[2]
                        break
                # Only consider unserved children who are currently waiting
                if waiting_place:
                     unserved_children[child] = {
                         'place': waiting_place,
                         'allergic': self.child_allergy.get(child, False) # Default to not allergic if status unknown
                     }

        # 2. If no children are unserved, the heuristic is 0 (goal state).
        if not unserved_children:
            return 0

        # 3. Count the total number of unserved children (N_unserved).
        N_allergic = sum(1 for c_info in unserved_children.values() if c_info['allergic'])
        N_not_allergic = sum(1 for c_info in unserved_children.values() if not c_info['allergic'])
        N_unserved = N_allergic + N_not_allergic

        h = N_unserved # Cost for the final 'serve' action for each child

        # 4. Count sandwiches in kitchen and on trays.
        num_gf_kitchen = sum(1 for s in self.all_sandwiches if match("(at_kitchen_sandwich " + s + ")", "at_kitchen_sandwich", s) and match("(no_gluten_sandwich " + s + ")", "no_gluten_sandwich", s))
        num_reg_kitchen = sum(1 for s in self.all_sandwiches if match("(at_kitchen_sandwich " + s + ")", "at_kitchen_sandwich", s) and not match("(no_gluten_sandwich " + s + ")", "no_gluten_sandwich", s))
        num_gf_ontray = sum(1 for s in self.all_sandwiches if any(match(fact, "ontray", s, "*") for fact in state) and match("(no_gluten_sandwich " + s + ")", "no_gluten_sandwich", s))
        num_reg_ontray = sum(1 for s in self.all_sandwiches if any(match(fact, "ontray", s, "*") for fact in state) and not match("(no_gluten_sandwich " + s + ")", "no_gluten_sandwich", s))

        # 5. Add cost for 'make' actions.
        # Needed_to_Make_GF: Number of GF sandwiches needed that are not yet in kitchen or on tray.
        Needed_GF_Total = N_allergic
        Avail_GF_Total = num_gf_kitchen + num_gf_ontray
        Needed_to_Make_GF = max(0, Needed_GF_Total - Avail_GF_Total)

        # Needed_to_Make_Reg: Number of Reg sandwiches needed that are not yet in kitchen or on tray.
        Needed_Reg_Total = N_not_allergic
        Avail_Reg_Total = num_reg_kitchen + num_reg_ontray
        Needed_to_Make_Reg = max(0, Needed_Reg_Total - Avail_Reg_Total)

        h += Needed_to_Make_GF + Needed_to_Make_Reg # Cost for make

        # 6. Add cost for 'put_on_tray' actions.
        # Needed_GF_on_Tray: Number of GF sandwiches needed that are not yet on tray.
        Needed_GF_on_Tray = max(0, N_allergic - num_gf_ontray)
        # Needed_Reg_on_Tray: Number of Reg sandwiches needed that are not yet on tray.
        Needed_Reg_on_Tray = max(0, N_not_allergic - num_reg_ontray)

        h += Needed_GF_on_Tray + Needed_Reg_on_Tray # Cost for put_on_tray

        # 7. Identify places needing trays.
        places_with_waiting_children = {c_info['place'] for c_info in unserved_children.values()}
        places_with_trays = {get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")}

        # 8. Count the number of places where children are waiting but no tray is present.
        places_needing_trays = places_with_waiting_children - places_with_trays

        # 9. Add cost for 'move_tray' actions.
        h += len(places_needing_trays)

        return h
