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
    # Ensure the number of parts matches the number of args if args are not wildcards
    if len(parts) != len(args) and '*' not in args:
        return False
    # Check if each part matches the corresponding arg (with fnmatch for wildcards)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of unserved children and estimates the steps needed
    to get a suitable sandwich to each child's location and serve them.
    The estimate is based on the minimum number of make, put_on_tray, move_tray,
    and serve actions required, aggregated across all unserved children.

    # Assumptions
    - There are always enough ingredients (bread, content) and sandwich slots (`notexist`)
      to make any required sandwich type (regular or gluten-free), provided `notexist`
      predicates exist for the needed number of sandwiches.
    - There is always at least one tray available somewhere to be moved if needed.
    - A single tray can carry multiple sandwiches.
    - The cost of moving a tray is 1, regardless of distance.
    - The heuristic sums the estimated costs for different stages (making, putting, moving, serving)
      assuming resources flow through the pipeline (ingredients -> kitchen sandwich -> on tray -> at location -> served).
      It counts the *number of items* needing processing at each stage (sandwiches to make, sandwiches to put on trays, places needing trays, children to serve).

    # Heuristic Initialization
    - Extract static facts: which children are allergic/not allergic.
    - Identify all children, sandwiches, trays, and places by examining facts in the initial state, goals, and static information.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all children who are waiting but not yet served. This count is `num_unserved`. If `num_unserved` is 0, the heuristic is 0.
    2. Count how many of these unserved children are allergic (`num_allergic_unserved`). (Note: This count is identified but not strictly used in the simplified additive cost calculation below, but could be used for a more complex heuristic).
    3. Count the number of sandwiches that have already been made (either in the kitchen or on a tray). This is `existing_total_sandwiches`.
    4. Count the number of gluten-free sandwiches that have already been made (`existing_gf_sandwiches`). (Note: This count is identified but not strictly used in the simplified additive cost calculation below).
    5. Estimate the number of new sandwiches that need to be made from ingredients. This is the total number of sandwiches required (`num_unserved`) minus the number already made (`existing_total_sandwiches`), minimum 0. This count is `cost_make`.
    6. Estimate the number of sandwiches that need to be put onto trays. This is the total number of sandwiches required (`num_unserved`) minus the number already on trays, minimum 0. This count is `cost_put_on_tray`.
    7. Estimate the number of tray movements needed. Identify all unique places where unserved children are waiting. Count how many trays are currently located at these required places. The number of tray movements needed is the number of unique required places minus the number of trays already at those places, minimum 0. This count is `cost_move_tray`.
    8. Estimate the number of serve actions needed. Each unserved child requires one serve action. This count is `cost_serve = num_unserved`.
    9. The total heuristic value is the sum of `cost_make`, `cost_put_on_tray`, `cost_move_tray`, and `cost_serve`. If the total is 0 but `num_unserved > 0`, return 1 to ensure the heuristic is non-zero for non-goal states.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Identify all objects of relevant types by examining arguments in facts
        self.all_children = set()
        self.all_sandwiches = set()
        self.all_trays = set()
        self.all_places = {'kitchen'} # kitchen is a constant place

        # Heuristic classification based on common predicates and argument positions
        for fact in task.initial_state | task.goals | self.static:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]

            # Check arguments based on predicate
            if predicate in {"served", "allergic_gluten", "not_allergic_gluten"} and len(args) == 1:
                self.all_children.add(args[0])
            elif predicate in {"waiting"} and len(args) == 2:
                 self.all_children.add(args[0])
                 self.all_places.add(args[1])
            elif predicate in {"at_kitchen_bread", "no_gluten_bread"} and len(args) == 1:
                 # bread-portions are not explicitly tracked by type in heuristic
                 pass
            elif predicate in {"at_kitchen_content", "no_gluten_content"} and len(args) == 1:
                 # content-portions are not explicitly tracked by type in heuristic
                 pass
            elif predicate in {"at_kitchen_sandwich", "no_gluten_sandwich", "notexist"} and len(args) == 1:
                 self.all_sandwiches.add(args[0])
            elif predicate in {"ontray"} and len(args) == 2:
                 self.all_sandwiches.add(args[0])
                 self.all_trays.add(args[1])
            elif predicate in {"at"} and len(args) == 2:
                 # Could be (at tray place) or (at ball room) etc. Assume (at tray place) in this domain
                 self.all_trays.add(args[0])
                 self.all_places.add(args[1])
            elif predicate in {"move_tray"} and len(args) == 3:
                 self.all_trays.add(args[0])
                 self.all_places.update(args[1:])
            elif predicate in {"make_sandwich", "make_sandwich_no_gluten"} and len(args) == 3:
                 self.all_sandwiches.add(args[0])
                 # bread and content args are not explicitly tracked by type
            elif predicate in {"put_on_tray"} and len(args) == 2:
                 self.all_sandwiches.add(args[0])
                 self.all_trays.add(args[1])
            elif predicate in {"serve_sandwich", "serve_sandwich_no_gluten"} and len(args) == 4:
                 self.all_sandwiches.add(args[0])
                 self.all_children.add(args[1])
                 self.all_trays.add(args[2])
                 self.all_places.add(args[3])


        # Ensure kitchen is always a place
        self.all_places.add('kitchen')


        # Store which children are allergic based on static facts
        self.allergic_children = {
            get_parts(fact)[1] for fact in self.static if match(fact, "allergic_gluten", "*")
        }
        # self.not_allergic_children = { # Not strictly needed for this heuristic logic
        #     get_parts(fact)[1] for fact in self.static if match(fact, "not_allergic_gluten", "*")
        # }


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children and their locations
        unserved_children_locations = {} # Map child to place
        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Iterate through all children that are goals
        children_in_goals = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        for child in children_in_goals:
            if child not in served_children_in_state:
                 # Find where this unserved child is waiting
                 waiting_place = None
                 for fact in state:
                     if match(fact, "waiting", child, "*"):
                         waiting_place = get_parts(fact)[2]
                         break
                 # Only consider children who are still waiting (they must be waiting to be served)
                 if waiting_place:
                     unserved_children_locations[child] = waiting_place

        num_unserved = len(unserved_children_locations)

        # If no children are unserved, the goal is reached.
        if num_unserved == 0:
            return 0

        # Count allergic unserved children (identified but not used in this simplified heuristic)
        # num_allergic_unserved = len([child for child in unserved_children_locations if child in self.allergic_children])

        # 2. Count existing sandwiches (made)
        existing_sandwiches_kitchen = {s for s in self.all_sandwiches if f"(at_kitchen_sandwich {s})" in state}
        existing_sandwiches_ontray = {s for s in self.all_sandwiches if any(f"(ontray {s} {t})" in state for t in self.all_trays)}
        existing_total_sandwiches = existing_sandwiches_kitchen | existing_sandwiches_ontray

        # Count existing GF sandwiches (made) (identified but not used in this simplified heuristic)
        # existing_gf_sandwiches = {s for s in existing_total_sandwiches if f"(no_gluten_sandwich {s})" in state}

        # 3. Estimate sandwiches to make
        # We need num_unserved total sandwiches.
        # Cost for making sandwiches = total sandwiches needed - existing total sandwiches
        cost_make = max(0, num_unserved - len(existing_total_sandwiches))

        # 4. Estimate sandwiches to put on trays
        sandwiches_on_trays = {s for s in self.all_sandwiches if any(f"(ontray {s} {t})" in state for t in self.all_trays)}
        sandwiches_to_put_on_trays = max(0, num_unserved - len(sandwiches_on_trays))
        cost_put_on_tray = sandwiches_to_put_on_trays

        # 5. Estimate tray movements
        places_with_unserved = set(unserved_children_locations.values())
        trays_at_required_places = {t for t in self.all_trays for p in places_with_unserved if f"(at {t} {p})" in state}
        num_places_needing_tray = len(places_with_unserved) - len(trays_at_required_places)
        cost_move_tray = max(0, num_places_needing_tray) # Need at least one tray per place

        # 6. Estimate serve actions
        cost_serve = num_unserved

        # Total heuristic cost
        total_cost = cost_make + cost_put_on_tray + cost_move_tray + cost_serve

        # Ensure heuristic is 0 only at goal
        if num_unserved == 0:
             return 0
        else:
             # Ensure heuristic is non-zero if not goal
             return max(1, total_cost)

