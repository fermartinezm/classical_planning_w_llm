from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts represented as strings

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure fact has enough parts to match args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def is_gluten_free_sandwich(state, sandwich_obj):
    """Check if a specific sandwich object is marked as gluten-free in the state."""
    return f"(no_gluten_sandwich {sandwich_obj})" in state

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnack domain.

    Estimates the number of actions needed to serve all children.
    The heuristic is the sum of estimated costs for the following stages:
    1. Making sandwiches that are needed but don't exist.
    2. Putting sandwiches (both existing in kitchen and newly made) onto trays.
    3. Moving trays to the locations where unserved children are waiting.
    4. Serving the unserved children.

    This heuristic is not admissible but aims to guide a greedy search efficiently
    by prioritizing states where necessary resources (sandwiches, trays) are closer
    to being in the right place for serving.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children, allergy info,
        and initial waiting locations from the task's static facts and goals.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal children (those who need to be served)
        # Assuming goal facts are always of the form (served ?c)
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == "served"}

        # Extract allergy information from static facts
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "allergic_gluten", "*")
        }

        # Extract initial waiting locations from static facts
        # Assuming initial waiting facts define where children need to be served
        # The domain does not seem to have actions changing the 'waiting' predicate,
        # so we treat the initial waiting location as the target serving location.
        self.waiting_children_locations = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in static_facts
            if match(fact, "waiting", "*", "*")
        }

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 1. Identify unserved children and their needs/locations
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = self.goal_children - served_children

        # If all goal children are served, the heuristic is 0
        if not unserved_children:
            return 0

        num_unserved = len(unserved_children)
        num_unserved_gf = sum(1 for c in unserved_children if c in self.allergic_children)
        num_unserved_reg = num_unserved - num_unserved_gf

        # Identify distinct places where unserved children are waiting
        # These are the places where trays are ultimately needed
        waiting_places = {self.waiting_children_locations[c] for c in unserved_children}

        # 2. Identify available sandwiches and their status
        sandwiches_in_kitchen = {s for fact in state if match(fact, "at_kitchen_sandwich", s)}
        sandwiches_ontray = {s for fact in state if match(fact, "ontray", s, "*")}
        available_sandwiches = sandwiches_in_kitchen | sandwiches_ontray

        # Determine which available sandwiches are gluten-free
        gf_sandwiches_available = {s for s in available_sandwiches if is_gluten_free_sandwich(state, s)}
        reg_sandwiches_available = available_sandwiches - gf_sandwiches_available

        # 3. Calculate sandwiches that still need to be made
        # We need enough GF sandwiches for allergic children and enough total
        # sandwiches for all children. Prioritize using available sandwiches.
        needed_to_make_gf = max(0, num_unserved_gf - len(gf_sandwiches_available))
        # Regular sandwiches can serve non-allergic children.
        # Total sandwiches needed is num_unserved.
        # Total available sandwiches is len(available_sandwiches).
        # Sandwiches that still need to be made (either GF or Reg) is the total needed minus total available.
        # A simpler approach for non-allergic: they can take any sandwich.
        # Let's count how many regular sandwiches are needed after satisfying GF needs.
        # Total sandwiches needed = num_unserved
        # Total sandwiches available = len(available_sandwiches)
        # Sandwiches needed = num_unserved
        # Sandwiches available = len(available_sandwiches)
        # Sandwiches that need making = max(0, num_unserved - len(available_sandwiches))
        # This doesn't distinguish GF/Reg correctly.
        # Let's stick to the previous logic: need num_unserved_gf GF, have len(gf_sandwiches_available). Need needed_to_make_gf GF.
        # Need num_unserved_reg Reg. Have len(reg_sandwiches_available). But Reg can also use surplus GF.
        # A better count for needed_to_make_reg:
        # Total sandwiches needed = num_unserved
        # Total sandwiches available = len(available_sandwiches)
        # Sandwiches that must be GF = num_unserved_gf
        # Sandwiches that can be Reg (or surplus GF) = num_unserved_reg
        # Sandwiches available that are GF = len(gf_sandwiches_available)
        # Sandwiches available that are Reg = len(reg_sandwiches_available)
        # Sandwiches needed that must be GF and aren't available GF = needed_to_make_gf
        # Sandwiches needed that can be Reg and aren't available (either Reg or surplus GF)
        # Surplus GF = max(0, len(gf_sandwiches_available) - num_unserved_gf)
        # Available for Reg needs = len(reg_sandwiches_available) + Surplus GF
        # Needed to make Reg = max(0, num_unserved_reg - Available for Reg needs)
        # This is getting complicated. Let's simplify back to the additive model:
        # Cost to make GF = needed_to_make_gf
        # Cost to make Reg = needed_to_make_reg (assuming no surplus GF helps here in the heuristic count)
        # This is a relaxation.
        needed_to_make_gf = max(0, num_unserved_gf - len(gf_sandwiches_available))
        needed_to_make_reg = max(0, num_unserved_reg - len(reg_sandwiches_available)) # This might overestimate if surplus GF exists
        # Let's use the simpler count: total sandwiches needed vs total available
        total_sandwiches_needed = num_unserved
        total_sandwiches_available = len(available_sandwiches)
        needed_to_make_any = max(0, total_sandwiches_needed - total_sandwiches_available)
        # This doesn't respect GF constraint. Let's go back to the per-type count.
        # The simplest additive count:
        needed_to_make_gf = max(0, num_unserved_gf - len(gf_sandwiches_available))
        needed_to_make_reg = max(0, num_unserved_reg - len(reg_sandwiches_available)) # This is the most straightforward additive component
        cost_make = needed_to_make_gf + needed_to_make_reg # Cost is 1 per sandwich made

        # 4. Calculate sandwiches currently in the kitchen that need to be put on a tray
        # This includes existing kitchen sandwiches and those that will be newly made
        num_at_kitchen_sandwiches = len(sandwiches_in_kitchen)
        # Every sandwich that is or will be in the kitchen needs a put_on_tray action
        cost_put_on_tray = num_at_kitchen_sandwiches + needed_to_make_gf + needed_to_make_reg

        # 5. Calculate trays that still need to move to a waiting location
        # Count trays currently at any of the waiting places
        trays_at_waiting_places = {
            t for fact in state if match(fact, "at", t, "*") and get_parts(fact)[2] in waiting_places
        }
        # We need at least one tray at each distinct waiting place.
        # The cost is the number of waiting places that don't currently have a tray.
        cost_move_tray = max(0, len(waiting_places) - len(trays_at_waiting_places))

        # 6. Calculate children that still need to be served
        # Each unserved child needs one serve action
        cost_serve = num_unserved

        # Total heuristic is the sum of costs for these sequential stages
        # This is an additive heuristic summing up estimated costs for different subgoals/bottlenecks.
        total_cost = cost_make + cost_put_on_tray + cost_move_tray + cost_serve

        return total_cost

