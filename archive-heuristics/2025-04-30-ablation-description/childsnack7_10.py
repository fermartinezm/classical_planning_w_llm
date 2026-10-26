from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve all
    unserved children. It does this by counting the "deficits" at different
    stages of the sandwich delivery pipeline: making sandwiches, putting them
    on trays, moving trays to children's locations, and finally serving the
    sandwiches. The total heuristic is the sum of the number of unserved
    children (representing the final 'serve' action needed for each) plus the
    number of sandwiches of each type (regular/gluten-free) that are missing
    at each preceding stage (kitchen-made, on-tray, at-location) relative to
    the total demand from unserved children.

    # Assumptions
    - The goal is to serve all children specified in the problem instance.
    - Ingredients (bread and content portions) are assumed to be sufficient
      if needed to make a sandwich for a solvable problem; the heuristic
      only counts the 'make_sandwich' action itself, not ingredient availability.
    - Each unserved child requires exactly one sandwich of the correct type
      (gluten-free if allergic, regular otherwise).
    - A tray move action can potentially bring multiple sandwiches to a location,
      but the heuristic counts the *number of sandwiches* that need to arrive,
      which is a lower bound on move actions needed for delivery.
    - A put-on-tray action applies to one sandwich at a time.
    - A make-sandwich action creates one sandwich.

    # Heuristic Initialization
    The heuristic constructor extracts static information from the task:
    - Which children are allergic to gluten and which are not.
    - The waiting place for each child.
    - A set of all possible places mentioned in the initial state or static facts.
    This information is used to determine the type of sandwich needed by each
    child and where it needs to be delivered.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic calculates the total cost as the sum of costs for each stage
    of the delivery process for all unserved children:

    1.  **Identify Unserved Children:** Count the total number of children who
        are not yet marked as `(served ?c)`. Separate this count into
        `unserved_reg` (non-allergic) and `unserved_gf` (allergic).
        The base heuristic cost is `unserved_reg + unserved_gf` (representing
        the final 'serve' action for each).

    2.  **Count Available Sandwiches:** Iterate through the current state to
        determine the number of regular and gluten-free sandwiches available at
        different stages:
        - `avail_reg_kitchen`, `avail_gf_kitchen`: Sandwiches `(at_kitchen_sandwich ?s)`.
        - `avail_reg_ontray`, `avail_gf_ontray`: Sandwiches `(ontray ?s ?t)` anywhere.
        - `avail_reg_at_place[p]`, `avail_gf_at_place[p]`: Sandwiches `(ontray ?s ?t)`
          where the tray `?t` is `(at ?t p)`. This requires mapping trays to their
          current locations.

    3.  **Calculate Deficits at Each Stage:** Compare the total demand from
        unserved children of each type (`unserved_reg`, `unserved_gf`) with the
        available supply at successive stages of the pipeline.
        - **Move Deficit (`D_move`):** The number of sandwiches of each type
          that need to be moved to children's locations. This is the total
          number of unserved children of a type minus the number of sandwiches
          of that type already available on trays at *any* child's waiting place.
          `D_move_reg = max(0, unserved_reg - sum(avail_reg_at_place.values()))`
          `D_move_gf = max(0, unserved_gf - sum(avail_gf_at_place.values()))`
          Cost += `D_move_reg + D_move_gf`.

        - **Put-on-Tray Deficit (`D_put`):** The number of sandwiches of each type
          that need to be put on trays. This is the total number of unserved
          children of a type minus the number of sandwiches of that type already
          available on trays anywhere.
          `D_put_reg = max(0, unserved_reg - avail_reg_ontray)`
          `D_put_gf = max(0, unserved_gf - avail_gf_ontray)`
          Cost += `D_put_reg + D_put_gf`.

        - **Make Deficit (`D_make`):** The number of sandwiches of each type
          that need to be made. This is the total number of unserved children
          of a type minus the number of sandwiches of that type already made
          (either at the kitchen or already on trays).
          `D_make_reg = max(0, unserved_reg - (avail_reg_ontray + avail_reg_kitchen))`
          `D_make_gf = max(0, unserved_gf - (avail_gf_ontray + avail_gf_kitchen))`
          Cost += `D_make_reg + D_make_gf`.

    4.  **Total Heuristic:** The sum of the base cost (serve actions) and the
        deficits at the move, put, and make stages.
        `h = (unserved_reg + unserved_gf) + (D_move_reg + D_move_gf) + (D_put_reg + D_put_gf) + (D_make_reg + D_make_gf)`

    This heuristic is admissible because each unit counted in the deficits
    represents a distinct, necessary step (make, put, move) for a sandwich
    to reach a state where it can eventually be served, and the final
    `unserved` count represents the necessary serve actions. The counts
    at different stages are cumulative deficits towards the final goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children,
        their waiting places, and all possible locations.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Also need initial state to find all places

        # Extract child types and waiting places from static facts
        self.allergic_children = {get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")}
        self.not_allergic_children = {get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")}
        self.waiting_places = {get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "waiting", "*", "*")}

        # Identify all possible places from static facts and initial state
        places_from_waiting = set(self.waiting_places.values())
        places_from_initial_at = {get_parts(fact)[2] for fact in initial_state if match(fact, "at", "*", "*")}
        self.all_places = places_from_waiting | places_from_initial_at
        # Ensure 'kitchen' is included if not explicitly in initial 'at' or waiting places
        # (It's a constant, but good to be sure it's in our place list if needed)
        self.all_places.add('kitchen')


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        @param node: The search node containing the current state.
        @return: The estimated cost (heuristic value) to reach a goal state.
        """
        state = node.state

        # 1. Identify Unserved Children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_reg = 0
        unserved_gf = 0
        all_children = self.allergic_children | self.not_allergic_children

        for child in all_children:
            if child not in served_children:
                if child in self.allergic_children:
                    unserved_gf += 1
                else: # child in self.not_allergic_children
                    unserved_reg += 1

        # If no children are unserved, the goal is reached, heuristic is 0.
        if unserved_reg == 0 and unserved_gf == 0:
            return 0

        # 2. Count Available Sandwiches at different stages
        gf_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        kitchen_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        ontray_facts = [fact for fact in state if match(fact, "ontray", "*", "*")]
        tray_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")}

        # Sandwiches available at the kitchen
        avail_reg_kitchen = len({s for s in kitchen_sandwiches_in_state if s not in gf_sandwiches_in_state})
        avail_gf_kitchen = len({s for s in kitchen_sandwiches_in_state if s in gf_sandwiches_in_state})

        # Sandwiches available on trays anywhere
        avail_reg_ontray = 0
        avail_gf_ontray = 0

        # Sandwiches available on trays at each specific place
        avail_reg_at_place = {p: 0 for p in self.all_places}
        avail_gf_at_place = {p: 0 for p in self.all_places}

        for fact in ontray_facts:
            parts = get_parts(fact)
            if len(parts) == 3: # Ensure fact is (ontray s t)
                _, s, t = parts
                is_gf = s in gf_sandwiches_in_state

                if is_gf:
                    avail_gf_ontray += 1
                else:
                    avail_reg_ontray += 1

                # Check if the tray's location is known and update place counts
                if t in tray_locations:
                    place = tray_locations[t]
                    if place in self.all_places: # Ensure place is one we care about
                         if is_gf:
                            avail_gf_at_place[place] += 1
                         else:
                            avail_reg_at_place[place] += 1


        # 3. Calculate Deficits at Each Stage

        # Deficit for moving sandwiches to the correct places
        # Number of sandwiches needed at places = total unserved children
        # Number of sandwiches available at places = sum of avail_at_place counts
        deficit_move_reg = max(0, unserved_reg - sum(avail_reg_at_place.values()))
        deficit_move_gf = max(0, unserved_gf - sum(avail_gf_at_place.values()))

        # Deficit for putting sandwiches on trays
        # Number of sandwiches needed on trays = total unserved children
        # Number of sandwiches available on trays = avail_ontray counts
        deficit_put_reg = max(0, unserved_reg - avail_reg_ontray)
        deficit_put_gf = max(0, unserved_gf - avail_gf_ontray)

        # Deficit for making sandwiches
        # Number of sandwiches needed made = total unserved children
        # Number of sandwiches available made = avail_ontray + avail_kitchen counts
        deficit_make_reg = max(0, unserved_reg - (avail_reg_ontray + avail_reg_kitchen))
        deficit_make_gf = max(0, unserved_gf - (avail_gf_ontray + avail_gf_kitchen))

        # 4. Total Heuristic
        # Sum of unserved children (serve action) + deficits at each preceding stage
        total_cost = (unserved_reg + unserved_gf) + \
                     (deficit_move_reg + deficit_move_gf) + \
                     (deficit_put_reg + deficit_put_gf) + \
                     (deficit_make_reg + deficit_make_gf)

        return total_cost

