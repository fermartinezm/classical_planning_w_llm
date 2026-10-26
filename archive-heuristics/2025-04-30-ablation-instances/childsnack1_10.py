from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings.
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
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
    # The fact must have at least as many parts as the pattern has arguments
    if len(parts) < len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children.
    It sums up the estimated minimum number of 'make sandwich', 'put on tray',
    'move tray', and 'serve sandwich' actions needed to satisfy the goal conditions.
    It considers gluten allergies and available sandwich types.

    # Assumptions
    - Ingredients for making sandwiches are available in the kitchen if the corresponding
      bread/content objects exist in the initial state (though not explicitly checked
      in the heuristic beyond counting 'notexist' sandwiches).
    - Trays have unlimited capacity for sandwiches.
    - Trays can move directly between any two locations.
    - The heuristic assumes a simplified flow: make sandwich (if needed) -> put on tray
      (if needed) -> move tray to child's location (if needed) -> serve child.
    - Resource contention (e.g., multiple sandwiches needing the same tray/kitchen spot
      simultaneously) is ignored.
    - If the required number of sandwiches (considering GF needs and total) exceeds
      the total number of sandwich objects available or makeable, the problem is
      considered unsolvable from this state, and a large heuristic value is returned.

    # Heuristic Initialization
    - Identify all children who need to be served based on the goal conditions.
    - Identify which children are allergic to gluten based on static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are not yet served in the current state.
    2. Categorize unserved children into allergic and non-allergic groups, noting their waiting locations.
    3. If all children are served, the heuristic is 0.
    4. Count the number of unserved allergic children (`num_allergic_unserved`) and non-allergic children (`num_non_allergic_unserved`).
    5. Count the current state of sandwiches:
       - Number of gluten-free sandwiches already on trays (`ontray_gf`).
       - Number of non-gluten-free sandwiches already on trays (`ontray_non_gf`).
       - Number of gluten-free sandwiches in the kitchen (`kitchen_gf`).
       - Number of non-gluten-free sandwiches in the kitchen (`kitchen_non_gf`).
       - Number of sandwich objects that do not yet exist (`num_notexist`). These can potentially be made.
    6. Calculate the number of *new* gluten-free sandwiches that *must* be made to satisfy allergic children: `needed_new_gf = max(0, num_allergic_unserved - (ontray_gf + kitchen_gf))`.
    7. If `needed_new_gf` exceeds the total number of makeable sandwiches (`num_notexist`), the problem is likely unsolvable with the given objects; return a large value.
    8. Calculate the number of *new* non-gluten-free (or additional GF) sandwiches needed for non-allergic children, considering the remaining makeable sandwiches after accounting for GF needs: `needed_new_non_gf = max(0, num_non_allergic_unserved - (ontray_non_gf + kitchen_non_gf))`.
    9. Check if the combined need for new sandwiches (`needed_new_gf + needed_new_non_gf`) exceeds the total makeable sandwiches (`num_notexist`). If so, return a large value. A more precise check: `needed_new_non_gf` must be satisfiable by `num_notexist - needed_new_gf`. If `needed_new_non_gf > num_notexist - needed_new_gf`, return large value.
    10. Calculate the total number of sandwiches that need to be made: `total_needed_to_make = needed_new_gf + needed_new_non_gf`.
    11. Calculate the total number of sandwiches required to serve all unserved children: `total_needed_sandwiches = num_allergic_unserved + num_non_allergic_unserved`.
    12. Calculate the number of sandwiches that need to be moved from the kitchen (or made) onto a tray: `needed_put_on_tray = max(0, total_needed_sandwiches - (ontray_gf + ontray_non_gf))`.
    13. Identify the set of unique locations where unserved children are waiting.
    14. Identify the current locations of all trays.
    15. Count the number of distinct waiting locations (excluding the kitchen) where unserved children are waiting and no tray is currently present. This estimates the minimum number of tray movements required to get trays to where they are needed.
    16. The total heuristic value is the sum of:
        - The total number of unserved children (estimating the number of 'serve' actions).
        - The total number of sandwiches that need to be made (estimating 'make' actions).
        - The number of sandwiches that need to be put on a tray (estimating 'put_on_tray' actions).
        - The number of distinct waiting locations (not kitchen) that currently lack a tray (estimating 'move_tray' actions).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Children who need to be served (from goal conditions).
        - Children who are allergic to gluten (from static facts).
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Identify all children who are allergic to gluten.
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "allergic_gluten", "*")
        }

        # Identify all children who need to be served based on the goal.
        self.all_children_in_goal = {
            get_parts(goal)[1]
            for goal in self.goals
            if match(goal, "served", "*")
        }

        # Define a large value for unsolvable states
        self.UNSOLVABLE_COST = 1000 # Use a large integer

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify unserved children and their waiting locations
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children_set = self.all_children_in_goal - served_children

        # 3. If all children are served, the heuristic is 0.
        if not unserved_children_set:
            return 0

        # 2. Categorize unserved children and note locations
        unserved_allergic_locs = {}
        unserved_non_allergic_locs = {}
        waiting_facts = {fact for fact in state if match(fact, "waiting", "*", "*")}

        for child in unserved_children_set:
            # Find the waiting location for this child
            location = None
            for fact in waiting_facts:
                parts = get_parts(fact)
                if len(parts) == 3 and parts[1] == child and parts[0] == 'waiting':
                    location = parts[2]
                    break # Assuming a child waits at only one place

            if location is None:
                 # This indicates an issue with the state/problem definition if an unserved
                 # child in the goal is not in a waiting state. Treat as unsolvable.
                 return self.UNSOLVABLE_COST

            if child in self.allergic_children:
                unserved_allergic_locs[child] = location
            else:
                unserved_non_allergic_locs[child] = location

        # 4. Count unserved children by type
        num_allergic_unserved = len(unserved_allergic_locs)
        num_non_allergic_unserved = len(unserved_non_allergic_locs)

        # 5. Count existing sandwiches and their status
        ontray_sandwiches_list = [get_parts(fact) for fact in state if match(fact, "ontray", "*", "*")]
        kitchen_sandwiches_list = [get_parts(fact) for fact in state if match(fact, "at_kitchen_sandwich", "*")]
        notexist_sandwiches_list = [get_parts(fact) for fact in state if match(fact, "notexist", "*")]
        gf_sandwich_names = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        ontray_gf = sum(1 for parts in ontray_sandwiches_list if len(parts) > 1 and parts[1] in gf_sandwich_names)
        ontray_non_gf = len(ontray_sandwiches_list) - ontray_gf
        kitchen_gf = sum(1 for parts in kitchen_sandwiches_list if len(parts) > 1 and parts[1] in gf_sandwich_names)
        kitchen_non_gf = len(kitchen_sandwiches_list) - kitchen_gf
        num_notexist = len(notexist_sandwiches_list)

        # 6-10. Calculate sandwich needs and make/put costs
        needed_new_gf = max(0, num_allergic_unserved - (ontray_gf + kitchen_gf))

        # Check if enough makeable sandwiches exist for GF needs
        if needed_new_gf > num_notexist:
            return self.UNSOLVABLE_COST

        remaining_makeable = num_notexist - needed_new_gf
        needed_new_non_gf = max(0, num_non_allergic_unserved - (ontray_non_gf + kitchen_non_gf))

        # Check if remaining makeable sandwiches are enough for non-GF needs
        if needed_new_non_gf > remaining_makeable:
             return self.UNSOLVABLE_COST

        # Total sandwiches that need to be made
        # This is the sum of new GF needed and new non-GF needed from the remaining makeable pool
        total_needed_to_make = needed_new_gf + needed_new_non_gf

        # Total sandwiches required to serve all unserved children
        total_needed_sandwiches = num_allergic_unserved + num_non_allergic_unserved

        # Number of sandwiches currently on trays
        total_existing_ontray = ontray_gf + ontray_non_gf

        # Number of sandwiches that need to be put on a tray
        needed_put_on_tray = max(0, total_needed_sandwiches - total_existing_ontray)

        # 11-15. Calculate move tray costs
        waiting_places = set(unserved_allergic_locs.values()) | set(unserved_non_allergic_locs.values())
        tray_locations = {
            get_parts(fact)[2]
            for fact in state
            if match(fact, "at", "*", "*") and len(get_parts(fact)) > 1 and get_parts(fact)[1].startswith("tray")
        }

        # Locations needing a tray visit (unserved children waiting there, not kitchen, and no tray present)
        locations_needing_tray_visit = {
            p for p in waiting_places
            if p != 'kitchen' and p not in tray_locations
        }
        move_cost = len(locations_needing_tray_visit)

        # 16. Total heuristic
        h = (num_allergic_unserved + num_non_allergic_unserved) # serve actions
        h += total_needed_to_make # make actions
        h += needed_put_on_tray # put on tray actions
        h += move_cost # move tray actions to needed locations

        return h
