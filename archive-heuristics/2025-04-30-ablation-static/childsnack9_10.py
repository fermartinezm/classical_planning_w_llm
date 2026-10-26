from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and ensure it's a string
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # Should not happen with standard PDDL fact representation in state
         # Return split string as a fallback, though it indicates unexpected input
         return fact_str.split()
    return fact_str[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(args) > len(parts):
        return False
    # Use zip to handle cases where parts might be longer than args (e.g., extra arguments)
    # fnmatch handles wildcards like '*'
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the number of unserved children (for the final 'serve' action) and adds
    estimated costs for the necessary preceding steps: making sandwiches, putting
    sandwiches onto trays, and moving trays to the children's locations. The heuristic
    attempts to account for shared resources (sandwiches, trays, locations) by focusing
    on the number of "delivery pipelines" needed for children who don't already have
    a suitable sandwich ready at their location.

    # Assumptions
    - Ingredients (bread, content) are available in sufficient quantities in the kitchen
      to make any required sandwiches (this is a simplification for non-admissibility).
    - Trays can be moved between any places.
    - Any tray can be used for any sandwich.
    - Any suitable sandwich on a tray at a child's location can be used to serve that child.
    - Children remain in the 'waiting' state until they are 'served'.
    - The goal is solely to serve all children specified in the goal conditions.

    # Heuristic Initialization
    - Extract static facts like which children are allergic to gluten.
    - Store the task object to access goal conditions if needed (though the main logic
      relies on identifying unserved children from the state).

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify all children who are waiting but not yet served. If this set is empty,
        the state is a goal state, and the heuristic is 0. Otherwise, add the count
        of unserved children to the total cost (each needs a final 'serve' action).
    2.  For each unserved child, check if a suitable sandwich is already on a tray
        at their waiting location. Identify the subset of unserved children who
        *do not* have a suitable sandwich ready at their location. These children
        represent the "delivery pipelines" that need to be completed.
    3.  Determine the suitability requirements for the children identified in step 2
        (i.e., whether any of them need a gluten-free sandwich or a regular one).
    4.  Count suitable sandwiches currently available anywhere (in the kitchen or on any tray)
        that meet the suitability requirements determined in step 3.
    5.  Estimate the number of *new* sandwiches that need to be made: This is the
        number of children identified in step 2 minus the count from step 4, minimum 0.
        Add this count to the total cost (each needs a 'make' action).
    6.  Count the number of suitable sandwiches currently `at_kitchen_sandwich` that
        meet the suitability requirements determined in step 3.
    7.  Estimate the number of sandwiches that need to be put on a tray: This is the
        minimum of the number of children identified in step 2 and the total number
        of suitable sandwiches that are currently at the kitchen (from step 6) or
        will be made (from step 5). Add this count to the total cost (each needs a
        'put_on_tray' action).
    8.  If any sandwiches need to be put on a tray (step 7 > 0) and no tray is
        currently `at kitchen`, add 1 to the cost (for moving a tray to the kitchen).
    9.  Identify all unique locations where the children identified in step 2 are waiting.
    10. Count how many of these unique locations *do not* currently have a tray (`at ?t ?p`).
        Add this count to the total cost (each needs a 'move_tray' action to bring a tray there).
    11. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        super().__init__(task) # Call parent constructor

        # Identify allergic children from static facts
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "allergic_gluten", "*")
        }

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state
        cost = 0

        # Pre-calculate GF sandwiches in the current state for efficient suitability check
        gf_sandwiches_in_state = {
            get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")
        }

        def is_suitable(sandwich, child):
            """Checks if a sandwich is suitable for a child based on allergies."""
            is_gf = sandwich in gf_sandwiches_in_state
            child_is_allergic = child in self.allergic_children
            # Allergic child needs GF. Non-allergic child can have any (GF or regular).
            return is_gf if child_is_allergic else True

        # 1. Identify unserved children and their locations
        unserved_children_waiting_at = {} # {child: place}
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "waiting":
                child, place = parts[1], parts[2]
                if child not in served_children:
                    unserved_children_waiting_at[child] = place

        num_unserved = len(unserved_children_waiting_at)

        # Check for goal state (all children who were waiting are now served)
        # Given the domain structure, this implies all goal children are served.
        if num_unserved == 0:
             return 0

        # If we are here, it's not a goal state. Add cost for final serve actions.
        cost += num_unserved

        # Find sandwiches currently on trays and their locations
        sandwiches_on_trays_and_loc = {} # {sandwich: tray_location}
        tray_locations = {}
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "at" and parts[1].startswith("tray"):
                 tray, loc = parts[1], parts[2]
                 tray_locations[tray] = loc

        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1], get_parts(fact)[2]
                if t in tray_locations:
                    sandwiches_on_trays_and_loc[s] = tray_locations[t]

        # 2. Identify children lacking a suitable sandwich on a tray at their location
        children_lacking_ready = {} # {child: place}
        locations_needing_delivery = set()

        for child, place in unserved_children_waiting_at.items():
            has_ready_sandwich = False
            for sandwich, s_loc in sandwiches_on_trays_and_loc.items():
                if s_loc == place and is_suitable(sandwich, child):
                    has_ready_sandwich = True
                    break # Found one ready sandwich for this child

            if not has_ready_sandwich:
                children_lacking_ready[child] = place
                locations_needing_delivery.add(place)

        num_children_lacking_ready_sandwich = len(children_lacking_ready)

        # If no children lack a ready sandwich, the remaining cost is just the serves (already added).
        if num_children_lacking_ready_sandwich == 0:
             return cost # Return the initial cost (which is num_unserved)

        # 3. Determine suitability requirements for the needy children
        any_allergic_needing = any(c in self.allergic_children for c in children_lacking_ready)
        any_non_allergic_needing = any(c not in self.allergic_children for c in children_lacking_ready)

        def is_suitable_for_any_needy(sandwich):
             is_gf = sandwich in gf_sandwiches_in_state
             if is_gf:
                 # GF is suitable if any allergic child needs it OR any non-allergic child needs it
                 return any_allergic_needing or any_non_allergic_needing
             else:
                 # Regular is suitable only if any non-allergic child needs it
                 return any_non_allergic_needing

        # 4. Count available suitable sandwiches anywhere (kitchen or on tray) for needy children
        available_sandwiches_anywhere = {
            get_parts(fact)[1]
            for fact in state
            if get_parts(fact)[0] in ["at_kitchen_sandwich", "ontray"]
        }
        suitable_available_anywhere = {
            s for s in available_sandwiches_anywhere
            if is_suitable_for_any_needy(s)
        }
        num_suitable_available_anywhere = len(suitable_available_anywhere)

        # 5. Estimate sandwich making cost
        sandwiches_to_make = max(0, num_children_lacking_ready_sandwich - num_suitable_available_anywhere)
        cost += sandwiches_to_make # 1 action per make_sandwich

        # 6. Count sandwiches at kitchen that are suitable for a needy child
        sandwiches_at_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        suitable_sandwiches_kitchen = {
             s for s in sandwiches_at_kitchen
             if is_suitable_for_any_needy(s)
        }
        num_suitable_kitchen = len(suitable_sandwiches_kitchen)

        # Count sandwiches that *will be* at kitchen and are suitable for a needy child
        # This is the ones currently there, plus the ones we plan to make.
        sandwiches_that_will_be_at_kitchen = num_suitable_kitchen + sandwiches_to_make

        # 7. Estimate putting on tray cost
        # We need to put enough sandwiches on trays for the children lacking a ready one.
        # These must come from the kitchen (either already there or just made).
        put_on_tray_needed = min(num_children_lacking_ready_sandwich, sandwiches_that_will_be_at_kitchen)
        cost += put_on_tray_needed # 1 action per put_on_tray

        # 8. Check if a tray is needed at the kitchen for the put_on_tray actions
        if put_on_tray_needed > 0:
             tray_at_kitchen = any(match(fact, "at", "tray*", "kitchen") for fact in state)
             if not tray_at_kitchen:
                 cost += 1 # 1 action to move a tray to the kitchen

        # 9. Estimate moving tray cost to locations needing delivery
        locations_with_tray = set(tray_locations.values())
        locations_to_move_tray_to = locations_needing_delivery - locations_with_tray
        cost += len(locations_to_move_tray_to) # 1 action per move_tray

        return cost
