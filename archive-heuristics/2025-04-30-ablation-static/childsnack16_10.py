# Helper function from Logistics example
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle potential malformed facts gracefully, maybe log a warning or skip
         # For this domain, facts are expected to be well-formed strings like '(predicate arg1 arg2)'
         # If it's not a string or doesn't look like a fact, return None or raise error
         # Assuming valid PDDL facts as strings:
         pass # Proceed with parsing

    # Remove outer parentheses and split by spaces
    parts = fact[1:-1].split()
    return parts

# Assuming heuristics.heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not available for standalone testing
import sys
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # print("Warning: heuristics.heuristic_base not found. Using dummy base class.", file=sys.stderr)
    class Heuristic:
        def __init__(self, task):
            self.task = task
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved
    children. It calculates the cost based on the current state of the sandwiches
    needed, prioritizing using sandwiches that are closer to being served
    (already on a tray at the child's location) and prioritizing gluten-free
    sandwiches for allergic children.

    # Assumptions
    - Each unserved child requires exactly one suitable sandwich to be served.
    - Action costs are uniform (cost 1 per action).
    - The heuristic assumes that trays, ingredients, and sandwich slots are
      available when needed for 'put_on_tray', 'move_tray', and 'make_sandwich'
      actions, up to the counts available in the state. It does not model
      complex resource contention beyond simple counts.
    - All children mentioned in the goal are initially in a 'waiting' state.
    - The problem is solvable, meaning enough ingredients and sandwich slots
      exist in total across the problem instance to make all necessary sandwiches.
      (The heuristic estimates based on *current* available ingredients/slots,
      but assumes any remaining needed sandwiches *can* be made eventually).

    # Heuristic Initialization
    The heuristic extracts static information from the task definition:
    - Which children are allergic to gluten.
    - Which children are not allergic to gluten.
    - Which bread portions are gluten-free.
    - Which content portions are gluten-free.
    - The set of children that need to be served according to the goal.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the estimated cost as follows:

    1.  **Identify Unserved Children:** Determine which children from the goal set
        have not yet been served in the current state. If none, heuristic is 0.
    2.  **Count Needed Sandwiches:** Count the number of unserved allergic children
        (who need gluten-free sandwiches) and unserved non-allergic children
        (who can take any sandwich).
    3.  **Categorize Available Sandwiches:** Count the number of suitable sandwiches
        available in different "readiness" pools based on their current state and
        location, distinguishing between gluten-free and regular sandwiches.
        A sandwich is "suitable" if it is gluten-free for an allergic child, or
        any type for a non-allergic child. Each sandwich/tray is counted in the
        cheapest applicable pool only once.
        -   **Pool 1 (Cost 1):** Suitable sandwiches already on a tray at the
            correct location for an unserved child. (Requires only the 'serve' action).
            Prioritize assigning gluten-free sandwiches from this pool to allergic children.
        -   **Pool 2 (Cost 2):** Suitable sandwiches already on a tray but at the
            kitchen or a wrong location, not used in Pool 1. (Requires 'move_tray' + 'serve').
        -   **Pool 3 (Cost 3):** Suitable sandwiches that are `at_kitchen_sandwich`,
            not used in Pool 1 or 2. (Requires 'put_on_tray' + 'move_tray' + 'serve').
        -   **Pool 4 (Cost 4):** Suitable sandwiches that do not yet exist (`notexist`)
            and must be made, limited by available ingredients and slots, not used
            in Pool 1, 2, or 3. (Requires 'make_sandwich' + 'put_on_tray' + 'move_tray' + 'serve').
    4.  **Assign Sandwiches to Needs:** Greedily assign available sandwiches from
        the cheapest pools (Pool 1 -> Pool 2 -> Pool 3 -> Pool 4) to fulfill the
        needed sandwiches. Prioritize fulfilling gluten-free needs first using
        available gluten-free sandwiches. Then, fulfill non-allergic needs using
        any remaining suitable sandwiches (regular or gluten-free) from the cheapest
        available pools.
    5.  **Calculate Total Cost:** Sum the cost associated with each assigned
        sandwich (1 for Pool 1, 2 for Pool 2, 3 for Pool 3, 4 for Pool 4). This sum
        represents the estimated number of actions to get all needed sandwiches
        served.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.

        @param task The planning task object.
        """
        super().__init__(task)

        # Extract static facts from the task
        self.allergic_children = {
            get_parts(fact)[1] for fact in task.static if get_parts(fact)[0] == 'allergic_gluten'
        }
        self.not_allergic_children = {
            get_parts(fact)[1] for fact in task.static if get_parts(fact)[0] == 'not_allergic_gluten'
        }
        self.bread_no_gluten = {
            get_parts(fact)[1] for fact in task.static if get_parts(fact)[0] == 'no_gluten_bread'
        }
        self.content_no_gluten = {
            get_parts(fact)[1] for fact in task.static if get_parts(fact)[0] == 'no_gluten_content'
        }

        # Extract children from the goal (assuming goal is conjunction of served predicates)
        self.goal_children = {
            get_parts(goal)[1] for goal in task.goals if get_parts(goal)[0] == 'served'
        }

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        @param node The current state node.
        @return The estimated cost to reach a goal state.
        """
        state = node.state

        # Parse dynamic facts from the current state
        state_parts = [get_parts(f) for f in state]

        served_children = {c for p, c in state_parts if p == 'served'}
        waiting_children = {c: loc for p, c, loc in state_parts if p == 'waiting'}
        at_kitchen_bread = {b for p, b in state_parts if p == 'at_kitchen_bread'}
        at_kitchen_content = {c for p, c in state_parts if p == 'at_kitchen_content'}
        at_kitchen_sandwich = {s for p, s in state_parts if p == 'at_kitchen_sandwich'}
        sandwiches_ontray = {s: t for p, s, t in state_parts if p == 'ontray'}
        trays_at_location = {t: loc for p, t, loc in state_parts if p == 'at'}
        sandwiches_notexist = {s for p, s in state_parts if p == 'notexist'}
        sandwiches_no_gluten = {s for p, s in state_parts if p == 'no_gluten_sandwich'}

        # Identify unserved children from the goal set
        unserved_children = {c for c in self.goal_children if c not in served_children}

        # If all goal children are served, the heuristic is 0
        if not unserved_children:
            return 0

        # Separate unserved children by allergy status
        unserved_allergic = {c for c in unserved_children if c in self.allergic_children}
        unserved_non_allergic = {c for c in unserved_children if c in self.not_allergic_children}

        # Count needed sandwiches
        N_ng_needed = len(unserved_allergic)
        N_reg_needed = len(unserved_non_allergic) # Non-allergic can take any sandwich

        # --- Count available sandwiches in different pools ---
        # Pool 1 (Cost 1: Serve): Suitable sandwich on tray at correct location
        avail_ng_cost1 = 0
        avail_reg_cost1 = 0
        used_s = set() # Track sandwiches used across all pools
        used_t = set() # Track trays used across all pools

        # Prioritize NG sandwiches at location for allergic children
        for child in unserved_allergic:
            loc = waiting_children.get(child) # Get child's location
            if loc is None: continue # Child not waiting? (Shouldn't happen in solvable states)
            found_match = False
            for s, t in sandwiches_ontray.items():
                if (s in sandwiches_no_gluten and
                    t in trays_at_location and trays_at_location[t] == loc and
                    s not in used_s and t not in used_t):
                    avail_ng_cost1 += 1
                    used_s.add(s)
                    used_t.add(t)
                    found_match = True
                    break # This sandwich/tray pair is used for one allergic child
            # Note: If multiple allergic children are at the same location,
            # this loop structure correctly assigns unique sandwiches/trays.

        # Use remaining sandwiches at location for non-allergic children
        for child in unserved_non_allergic:
            loc = waiting_children.get(child) # Get child's location
            if loc is None: continue # Child not waiting?
            found_match = False
            for s, t in sandwiches_ontray.items():
                if (t in trays_at_location and trays_at_location[t] == loc and
                    s not in used_s and t not in used_t): # Any sandwich
                    avail_reg_cost1 += 1
                    used_s.add(s)
                    used_t.add(t)
                    found_match = True
                    break # This sandwich/tray pair is used for one non-allergic child

        # Pool 2 (Cost 2: Move + Serve): Suitable sandwich on tray elsewhere
        avail_ng_cost2 = 0
        avail_reg_cost2 = 0

        # Prioritize NG sandwiches on trays elsewhere for remaining needs
        for s, t in sandwiches_ontray.items():
            if s not in used_s and t not in used_t:
                 # It's on a tray, not used for cost1. Cost is 2.
                if s in sandwiches_no_gluten:
                    avail_ng_cost2 += 1
                else:
                    avail_reg_cost2 += 1
                used_s.add(s)
                used_t.add(t)


        # Pool 3 (Cost 3: Put + Move + Serve): Suitable sandwich at kitchen
        avail_ng_cost3 = 0
        avail_reg_cost3 = 0

        # Prioritize NG sandwiches at kitchen for remaining needs
        for s in sandwiches_at_kitchen:
            if s not in used_s:
                if s in sandwiches_no_gluten:
                    avail_ng_cost3 += 1
                else:
                    avail_reg_cost3 += 1
                used_s.add(s)


        # Pool 4 (Cost 4: Make + Put + Move + Serve): Suitable sandwich needs making
        # Count ingredients and slots
        num_slots = len(sandwiches_notexist)
        num_bread_ng = len(at_kitchen_bread & self.bread_no_gluten)
        num_content_ng = len(at_kitchen_content & self.content_no_gluten)
        num_bread_reg = len(at_kitchen_bread) - num_bread_ng
        num_content_reg = len(at_kitchen_content) - num_content_ng

        # How many NG sandwiches can we make? Limited by NG ingredients and slots.
        can_make_ng = min(num_bread_ng, num_content_ng, num_slots)
        # How many Reg sandwiches can we make? Limited by Reg ingredients and remaining slots.
        can_make_reg = min(num_bread_reg, num_content_reg, num_slots - can_make_ng)

        avail_ng_cost4 = can_make_ng
        avail_reg_cost4 = can_make_reg


        # --- Assign sandwiches from pools to needs, cheapest first ---
        H = 0
        needed_ng = N_ng_needed
        needed_reg = N_reg_needed

        # Serve NG needs (prioritize NG sandwiches)
        use = min(needed_ng, avail_ng_cost1)
        H += use * 1
        needed_ng -= use

        use = min(needed_ng, avail_ng_cost2)
        H += use * 2
        needed_ng -= use
        avail_ng_cost2 -= use # Consume from pool

        use = min(needed_ng, avail_ng_cost3)
        H += use * 3
        needed_ng -= use
        avail_ng_cost3 -= use # Consume from pool

        use = min(needed_ng, avail_ng_cost4)
        H += use * 4
        needed_ng -= use
        avail_ng_cost4 -= use # Consume from pool


        # Serve Reg needs (can use Reg or remaining NG sandwiches)
        use = min(needed_reg, avail_reg_cost1)
        H += use * 1
        needed_reg -= use

        use = min(needed_reg, avail_reg_cost2)
        H += use * 2
        needed_reg -= use
        avail_reg_cost2 -= use # Consume from pool

        # Use leftover NG cost2 for Reg needs
        use = min(needed_reg, avail_ng_cost2)
        H += use * 2
        needed_reg -= use
        avail_ng_cost2 -= use # Consume from pool

        use = min(needed_reg, avail_reg_cost3)
        H += use * 3
        needed_reg -= use
        avail_reg_cost3 -= use # Consume from pool

        # Use leftover NG cost3 for Reg needs
        use = min(needed_reg, avail_ng_cost3)
        H += use * 3
        needed_reg -= use
        avail_ng_cost3 -= use # Consume from pool

        use = min(needed_reg, avail_reg_cost4)
        H += use * 4
        needed_reg -= use
        avail_reg_cost4 -= use # Consume from pool

        # Use leftover NG cost4 for Reg needs
        use = min(needed_reg, avail_ng_cost4)
        H += use * 4
        needed_reg -= use
        avail_ng_cost4 -= use # Consume from pool

        # If needed_ng or needed_reg is still > 0 here, it means we couldn't
        # make enough sandwiches even with available ingredients/slots.
        # For a solvable problem, this shouldn't happen if the initial state
        # plus available slots allow making all goal sandwiches.
        # We assume solvable problems have finite heuristic values.
        # Any remaining needed sandwiches would imply the problem is unsolvable
        # from this state with the *current* resources, but we assume the
        # problem instance as a whole is solvable. We can add a large cost
        # or assume they can eventually be made (cost 4). The current logic
        # implicitly handles this by using up the cost4 pool. If needed_ng/reg
        # are still > 0, it means even the makeable ones weren't enough.
        # This might indicate an issue with ingredient/slot counting or problem structure.
        # For simplicity and non-admissibility, we can just add the max cost for any remaining.
        # H += needed_ng * 4 # Add cost for any NG sandwiches we couldn't make
        # H += needed_reg * 4 # Add cost for any Reg sandwiches we couldn't make
        # This shouldn't be necessary if the problem is solvable and the counts are right.

        return H
