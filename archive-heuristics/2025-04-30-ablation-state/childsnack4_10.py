from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    This heuristic estimates the number of actions required to serve all children.
    It sums the estimated minimum number of actions needed for each stage of the
    delivery pipeline (Make -> Put -> Move -> Serve) across all unserved children.

    # Assumptions
    - Each child needs exactly one sandwich.
    - All children must be served to reach the goal.
    - Sufficient bread, content, and 'notexist' sandwich slots are available to make needed sandwiches.
    - Trays can hold multiple sandwiches.
    - The cost of moving a tray is 1, regardless of distance.
    - The cost of each action type (make, put, move, serve) is 1.
    - Suitability of a sandwich for a child is based on allergy status (GF for allergic, Any for non-allergic).

    # Heuristic Initialization
    - Identify all children, their allergy status, and their waiting locations from static and initial state facts.
    - Identify all trays and places from initial state facts.
    - Identify all sandwiches from initial state facts (even if they don't exist yet).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic sums the estimated minimum number of actions required for each
    transition stage for the *total number of unserved children's needs*:
    1.  **Serve Actions:** Each unserved child requires one 'serve' action. Count the total number of unserved children (`N_unserved`).
    2.  **Make Actions:** Estimate the number of sandwiches that need to be made. This is the total number of unserved children minus the number of suitable sandwiches already existing (in kitchen or on trays). `max(0, N_unserved - N_avail_anywhere)`.
    3.  **Put Actions:** Estimate the number of sandwiches that need to be put on trays. This is the total number of unserved children minus the number of suitable sandwiches already on trays. `max(0, N_unserved - N_ontray_anywhere)`.
    4.  **Move Tray Actions:** Estimate the number of tray movements required to bring trays to locations where unserved children are waiting and need a delivery (i.e., no suitable sandwich is already on a tray at their location) and no tray is currently present. Count distinct locations `p` where this condition holds.

    The total heuristic value is the sum of the counts from steps 1, 2, 3, and 4.
    Note: N_avail_anywhere and N_ontray_anywhere count the *number of suitable sandwiches* available, not the number of children whose needs they can meet. This is a simplification for efficiency.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children,
        trays, and places.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        self.all_children = set()
        self.all_trays = set()
        self.all_places = set()
        self.all_sandwiches = set()

        self.child_allergy = {} # child -> True if allergic, False otherwise
        self.child_location = {} # child -> place

        # Extract objects and static/initial info
        all_facts = set(self.static_facts) | set(self.initial_state) | set(self.goals)
        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts
            predicate = parts[0]
            if predicate in ["allergic_gluten", "not_allergic_gluten", "served"]:
                if len(parts) > 1: self.all_children.add(parts[1])
                if predicate == "allergic_gluten" and len(parts) > 1: self.child_allergy[parts[1]] = True
                if predicate == "not_allergic_gluten" and len(parts) > 1: self.child_allergy[parts[1]] = False
            elif predicate == "waiting":
                if len(parts) > 2:
                    child, place = parts[1], parts[2]
                    self.all_children.add(child)
                    self.all_places.add(place)
                    self.child_location[child] = place
            elif predicate == "at":
                if len(parts) > 2:
                    obj, place = parts[1], parts[2]
                    if obj.startswith("tray"): self.all_trays.add(obj)
                    self.all_places.add(place)
            elif predicate == "ontray":
                if len(parts) > 2:
                    sandwich, tray = parts[1], parts[2]
                    self.all_sandwiches.add(sandwich)
                    self.all_trays.add(tray)
            elif predicate in ["at_kitchen_sandwich", "notexist", "no_gluten_sandwich"]:
                 if len(parts) > 1: self.all_sandwiches.add(parts[1])
            # Bread and content objects are not strictly needed for this heuristic's counts

        # Ensure kitchen is included if not mentioned in relevant facts
        self.all_places.add("kitchen")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h_cost = 0

        # 1. Count unserved children
        unserved_children = {c for c in self.all_children if f"(served {c})" not in state}
        n_unserved = len(unserved_children)

        if n_unserved == 0:
            return 0 # Goal reached

        # Cost for 'serve' actions (each unserved child needs one)
        h_cost += n_unserved

        # Count allergic and non-allergic unserved children globally
        n_allergic_unserved = sum(1 for c in unserved_children if self.child_allergy.get(c, False))
        n_non_allergic_unserved = n_unserved - n_allergic_unserved

        # Helper to check if a sandwich is suitable for *any* unserved child globally
        def is_suitable_for_any_unserved_global(sandwich, current_state):
            is_gf_s = f"(no_gluten_sandwich {sandwich})" in current_state
            if is_gf_s:
                return n_allergic_unserved > 0 or n_non_allergic_unserved > 0
            else: # Regular sandwich
                return n_non_allergic_unserved > 0

        # Count suitable sandwiches available anywhere (kitchen or on trays)
        suitable_sandwiches_avail = {s for s in self.all_sandwiches if (f"(at_kitchen_sandwich {s})" in state or any(f"(ontray {s} {t})" in state for t in self.all_trays)) and is_suitable_for_any_unserved_global(s, state)}
        n_avail_anywhere = len(suitable_sandwiches_avail)

        # Count suitable sandwiches available on trays anywhere
        suitable_sandwiches_ontray = {s for s in self.all_sandwiches if any(f"(ontray {s} {t})" in state for t in self.all_trays) and is_suitable_for_any_unserved_global(s, state)}
        n_ontray_anywhere = len(suitable_sandwiches_ontray)

        # 2. Count sandwiches needing make (Make actions)
        # Number of sandwiches that need to be made is the total needed (N_unserved)
        # minus those already available anywhere (kitchen or ontray)
        n_to_make = max(0, n_unserved - n_avail_anywhere)
        h_cost += n_to_make # Cost for 'make' actions

        # 3. Count sandwiches needing put on tray (Put actions)
        # Number of sandwiches that need to be put on trays is the total needed (N_unserved)
        # minus those already on trays
        n_to_put = max(0, n_unserved - n_ontray_anywhere)
        h_cost += n_to_put # Cost for 'put_on_tray' actions

        # 4. Count locations needing tray move (Move actions)
        n_locations_needing_tray_move = 0
        locations_with_unserved = {self.child_location[c] for c in unserved_children if c in self.child_location}

        # Pre-calculate allergic/non-allergic unserved counts per location
        allergic_unserved_at_location = {p: 0 for p in self.all_places}
        non_allergic_unserved_at_location = {p: 0 for p in self.all_places}
        for c in unserved_children:
            p = self.child_location.get(c)
            if p:
                if self.child_allergy.get(c, False):
                    allergic_unserved_at_location[p] += 1
                else:
                    non_allergic_unserved_at_location[p] += 1


        for p in locations_with_unserved:
            n_allergic_unserved_at_p = allergic_unserved_at_location[p]
            n_non_allergic_unserved_at_p = non_allergic_unserved_at_location[p]

            # Check if any unserved child at this location needs a delivery to this location
            # A delivery is needed if there is NO suitable sandwich on a tray *at* location p
            needs_delivery_to_p = False
            # Find if there exists a suitable sandwich on a tray *at* location p
            found_suitable_ontray_at_p = False
            for s in self.all_sandwiches:
                is_gf_s = f"(no_gluten_sandwich {s})" in state

                # Check if this sandwich 's' is suitable for *any* unserved child at location p
                is_suitable_for_any_unserved_at_p = False
                if is_gf_s:
                    if n_allergic_unserved_at_p > 0 or n_non_allergic_unserved_at_p > 0:
                        is_suitable_for_any_unserved_at_p = True
                else: # Regular sandwich
                    if n_non_allergic_unserved_at_p > 0:
                        is_suitable_for_any_unserved_at_p = True

                if is_suitable_for_any_unserved_at_p:
                    # Check if this sandwich is on a tray that is at location p
                    for t in self.all_trays:
                        if f"(ontray {s} {t})" in state and f"(at {t} {p})" in state:
                            found_suitable_ontray_at_p = True
                            break # Found a suitable sandwich on a tray at p
                if found_suitable_ontray_at_p:
                    break # Found a suitable sandwich on a tray at p

            if not found_suitable_ontray_at_p:
                # No suitable sandwich is on a tray at location p for any unserved child at p
                needs_delivery_to_p = True

            if needs_delivery_to_p:
                # Check if there is any tray already at location p
                tray_at_p = any(f"(at {t} {p})" in state for t in self.all_trays)
                if not tray_at_p:
                    # This location needs a tray moved to it
                    n_locations_needing_tray_move += 1

        h_cost += n_locations_needing_tray_move # Cost for 'move_tray' actions

        # 5. Serve actions cost is already added at the beginning (n_unserved)

        return h_cost
