from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved
    children. It sums the estimated cost for each unserved child independently,
    ignoring resource contention (sandwiches, trays). The cost for a child is
    estimated based on the "stage" of readiness of a suitable sandwich:
    - 1 action: Serve the child (this is the base cost for any unserved child).
    - Additional cost (0, 1, 2, or 3) depending on where a suitable sandwich is:
        - 0: If a suitable sandwich is already on a tray at the child's location.
        - 1: If a suitable sandwich is on a tray elsewhere (needs `move_tray`).
        - 2: If a suitable sandwich is at the kitchen (needs `put_on_tray` + `move_tray` if not at kitchen).
        - 3: If no suitable sandwich exists anywhere (needs `make_sandwich` + `put_on_tray` + `move_tray` if not at kitchen).
    The `move_tray` action cost is only added if the child is not waiting at the kitchen.

    # Assumptions
    - All children in the goal state must be served.
    - The only way to serve a child is with a suitable sandwich on a tray at their location.
    - Gluten-allergic children require gluten-free sandwiches. Non-allergic children can have any sandwich.
    - Making a sandwich requires available ingredients and a 'notexist' sandwich object (this heuristic simplifies by assuming this is always possible if needed).
    - Putting a sandwich on a tray requires a tray at the kitchen (this heuristic simplifies by assuming a tray is available).
    - Moving a tray between any two places costs 1 action.
    - The kitchen is a place like any other for tray movement, but putting a sandwich on a tray only happens at the kitchen.
    - Unserved children in the goal are always in a 'waiting' state at some place.

    # Heuristic Initialization
    - Extract the set of children that need to be served from the goal conditions.
    - Extract the set of children who are allergic to gluten from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the set of children who are in the goal state but are not yet served in the current state. These are the unserved children.
    2. Initialize the total heuristic value `h` to 0.
    3. For each unserved child:
        a. Add 1 to `h` (representing the final 'serve' action).
        b. Find the child's waiting location `P` from the state facts.
        c. Determine if the child is allergic to gluten using the pre-calculated static information.
        d. Check if there is already a suitable sandwich `S` on a tray `T` that is located at the child's waiting place `P` (i.e., `(ontray S T)` and `(at T P)` are true in the state, and `S` is suitable for the child based on allergy status and `(no_gluten_sandwich S)` predicate in the state).
        e. If such a condition is met, the prerequisites for serving this child *at their location* are satisfied. No further cost is added for this child's sandwich delivery stages.
        f. If the condition in (d) is NOT met:
            i. Check if there is a suitable sandwich `S` on a tray `T` that is located *elsewhere* (i.e., `(ontray S T)` is true, `(at T P)` is false, and `S` is suitable).
            ii. If yes, this child's sandwich needs a tray movement. Add 1 to `h` (for `move_tray`).
            iii. If no suitable sandwich is on a tray anywhere:
                iv. Check if there is a suitable sandwich `S` located `at_kitchen_sandwich`.
                v. If yes, this child's sandwich needs to be put on a tray and then moved. Add 1 to `h` (for `put_on_tray`). Add 1 to `h` (for `move_tray`) *only if* the child is not waiting at the kitchen.
                vi. If no suitable sandwich exists anywhere (neither on tray nor at kitchen): This child's sandwich needs to be made, put on a tray, and moved. Add 1 to `h` (for `make_sandwich`), 1 (for `put_on_tray`), and 1 (for `move_tray`) *only if* the child is not waiting at the kitchen.
    4. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children and allergy status.
        """
        self.goal_children = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "served":
                self.goal_children.add(parts[1])

        self.allergic_children = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "allergic_gluten":
                self.allergic_children.add(parts[1])

        # We don't need not_allergic_gluten explicitly, as it's the complement
        # for children mentioned in static facts, or children not mentioned.
        # The logic only needs to know if a child *is* allergic.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # 1. Identify unserved children
        unserved_children = set()
        for child in self.goal_children:
            if f"(served {child})" not in state:
                unserved_children.add(child)

        # 3. For each unserved child
        for child in unserved_children:
            # a. Add 1 for the serve action
            h += 1

            # b. Find child's waiting place
            child_place = None
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "waiting" and parts[1] == child:
                    child_place = parts[2]
                    break
            if child_place is None:
                 # Should not happen in valid problem instances where goal children are waiting.
                 # If it happens, this child cannot be served. For a non-admissible heuristic,
                 # we can skip or add a large penalty. Skipping is simpler here.
                 continue

            # c. Check if child is allergic
            is_allergic = child in self.allergic_children

            # d. Check if suitable sandwich is on tray at child's place
            prerequisite_met = False
            suitable_sandwich_on_tray_elsewhere = False
            suitable_sandwich_at_kitchen = False

            # Find all suitable sandwiches that are on trays
            suitable_ontray_sandwiches = set()
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "ontray":
                    sandwich = parts[1]
                    tray = parts[2]
                    # Check if sandwich is suitable
                    is_gluten_free_sandwich = (f"(no_gluten_sandwich {sandwich})" in state)
                    is_suitable = (is_allergic and is_gluten_free_sandwich) or (not is_allergic)

                    if is_suitable:
                        suitable_ontray_sandwiches.add((sandwich, tray))

            # Check if any suitable on-tray sandwich is at the child's place
            for sandwich, tray in suitable_ontray_sandwiches:
                 if f"(at {tray} {child_place})" in state:
                     prerequisite_met = True
                     break # Found one that meets the prerequisite

            # e. If prerequisite is NOT met
            if not prerequisite_met:
                # f.i. Check if suitable sandwich is on tray elsewhere
                for sandwich, tray in suitable_ontray_sandwiches:
                    if f"(at {tray} {child_place})" not in state:
                        suitable_sandwich_on_tray_elsewhere = True
                        break # Found one on tray elsewhere

                # f.iii. If no suitable sandwich is on tray anywhere
                if not suitable_sandwich_on_tray_elsewhere:
                    # f.iv. Check if suitable sandwich is at kitchen
                    for fact in state:
                        parts = get_parts(fact)
                        if parts[0] == "at_kitchen_sandwich":
                            sandwich = parts[1]
                            # Check if sandwich is suitable
                            is_gluten_free_sandwich = (f"(no_gluten_sandwich {sandwich})" in state)
                            is_suitable = (is_allergic and is_gluten_free_sandwich) or (not is_allergic)
                            if is_suitable:
                                suitable_sandwich_at_kitchen = True
                                break # Found one at kitchen

                    # f.v. If suitable sandwich is at kitchen
                    if suitable_sandwich_at_kitchen:
                        h += 1 # Cost for put_on_tray
                        if child_place != "kitchen":
                            h += 1 # Cost for move_tray
                    # f.vi. If no suitable sandwich exists anywhere
                    else:
                        h += 1 # Cost for make_sandwich
                        h += 1 # Cost for put_on_tray
                        if child_place != "kitchen":
                            h += 1 # Cost for move_tray
                else: # f.ii. If suitable sandwich is on tray elsewhere
                    h += 1 # Cost for move_tray

        return h
