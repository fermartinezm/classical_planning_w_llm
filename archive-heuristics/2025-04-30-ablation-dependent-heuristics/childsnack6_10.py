# Import necessary modules
from collections import defaultdict

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple (predicate, obj1, obj2, ...)."""
    # Remove surrounding brackets and split by space
    parts = fact_string[1:-1].split()
    return tuple(parts)

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.
    """

    def __init__(self, task):
        """
        Initializes the heuristic with static information from the task.

        Heuristic Initialization:
            The constructor parses static facts to identify allergic/non-allergic
            children and their waiting places. It also identifies all children
            that need to be served from the goal facts.

        Args:
            task: The planning task object containing initial state, goals, and static facts.
        """
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.child_places = {} # child -> place
        self.all_children = set()

        # Parse static info
        for fact_string in task.static:
            fact = parse_fact(fact_string)
            if fact[0] == 'allergic_gluten':
                self.allergic_children.add(fact[1])
            elif fact[0] == 'not_allergic_gluten':
                self.not_allergic_children.add(fact[1])
            elif fact[0] == 'waiting':
                self.child_places[fact[1]] = fact[2]

        # Get all children from the goal facts
        for goal_fact_string in task.goals:
             fact = parse_fact(goal_fact_string)
             if fact[0] == 'served':
                 self.all_children.add(fact[1])

    def __call__(self, state):
        """
        Computes the domain-dependent heuristic value for the given state.

        Summary:
            The heuristic estimates the remaining effort by summing up the minimum
            number of actions required across all unserved children, considering the
            current state of sandwiches and trays. It counts the number of
            unserved children (serve actions), plus the number of sandwiches that
            still need to be put on a tray (put_on_tray actions), plus the number
            of sandwiches that still need to be made (make actions), plus the
            number of unique locations of unserved children that currently lack
            a tray (move actions).

        Assumptions:
            - The problem is solvable (sufficient ingredients and trays exist in total).
            - The heuristic does not attempt to match specific sandwiches to specific
              children or trays, but rather counts the total number of items/steps
              needed across all unserved children.
            - Gluten-free sandwiches can be served to non-allergic children.
            - The heuristic is non-admissible and designed for greedy best-first search.
            - The heuristic value is 0 only in goal states.
            - The heuristic value is finite for solvable states.

        Step-By-Step Thinking for Computing Heuristic:
            1. Identify all children who are currently unserved by checking the state
               against the set of all children from the goal and the 'served' predicate.
            2. If no children are unserved, the goal is reached, return 0.
            3. Count the total number of unserved children (`num_unserved_total`). This
               is a lower bound on the number of 'serve' actions needed. Initialize the
               heuristic component `h_serve` with this count.
            4. Iterate through the state facts to identify:
               - Sandwiches currently on trays (`ontray_sandwiches_set`).
               - Sandwiches currently in the kitchen (`kitchen_sandwiches_set`).
               - Sandwiches that do not yet exist (`notexist_sandwiches_set`).
               - Sandwiches currently marked as gluten-free (`no_gluten_sandwich`).
               - Trays and their current locations (`trays_at_places_state`).
            5. Calculate the number of sandwiches that have been made (either in kitchen or on tray).
               This is `num_made_sandwiches = len(ontray_sandwiches_set | kitchen_sandwiches_set)`.
            6. Calculate the number of sandwiches that still need to be made from ingredients
               to satisfy the unserved children. This is `num_need_make = max(0, num_unserved_total - num_made_sandwiches)`.
               Initialize the heuristic component `h_make` with this count.
            7. Count the total number of sandwiches that are currently on trays.
               This is `num_ontray_sandwiches = len(ontray_sandwiches_set)`.
            8. Calculate the number of sandwiches that still need to be moved from the kitchen
                onto a tray to satisfy the unserved children. This is `num_need_put_on_tray = max(0, num_unserved_total - num_ontray_sandwiches)`.
                Initialize the heuristic component `h_put_on_tray` with this count.
            9. Identify the set of unique places where unserved children are waiting (`waiting_places_of_unserved`).
            10. Identify the set of unique places where trays are currently located (`places_with_trays`).
            11. Calculate the number of places where unserved children are waiting but where no tray
                is currently present (`places_needing_tray = waiting_places_of_unserved - places_with_trays`).
            12. The number of tray moves needed is at least the number of these places.
                Initialize the heuristic component `h_move` with `len(places_needing_tray)`.
            13. The total heuristic is the sum of the components: `h_serve + h_put_on_tray + h_make + h_move`.

        Args:
            state: A frozenset of strings representing the current state facts.

        Returns:
            An integer representing the estimated number of actions to reach the goal.
        """
        # --- Step 1: Identify unserved children ---
        served_children = {parse_fact(f)[1] for f in state if parse_fact(f)[0] == 'served'}
        unserved_children = self.all_children - served_children

        # --- Step 2: Goal reached check ---
        if not unserved_children:
            return 0

        num_unserved_total = len(unserved_children)

        # --- Step 3: Heuristic component for serve actions ---
        h_serve = num_unserved_total

        # --- Step 4: Identify sandwiches and their states, and tray locations ---
        ontray_sandwiches_set = set() # set of sandwich names on trays
        kitchen_sandwiches_set = set() # set of sandwich names in kitchen
        notexist_sandwiches_set = set() # set of sandwich names that don't exist yet
        no_gluten_sandwiches_state = set() # set of sandwiches marked as no_gluten

        trays_at_places_state = defaultdict(set) # place -> set of trays

        for fact_string in state:
            fact = parse_fact(fact_string)
            if fact[0] == 'ontray':
                ontray_sandwiches_set.add(fact[1])
            elif fact[0] == 'at_kitchen_sandwich':
                kitchen_sandwiches_set.add(fact[1])
            elif fact[0] == 'notexist':
                notexist_sandwiches_set.add(fact[1])
            elif fact[0] == 'no_gluten_sandwich':
                no_gluten_sandwiches_state.add(fact[1])
            elif fact[0] == 'at':
                 trays_at_places_state[fact[2]].add(fact[1]) # place -> tray

        # --- Step 5-6: Heuristic component for make actions ---
        # Count sandwiches that are already made (in kitchen or on tray)
        num_made_sandwiches = len(ontray_sandwiches_set | kitchen_sandwiches_set)
        # Number of sandwiches that still need to be made
        h_make = max(0, num_unserved_total - num_made_sandwiches)

        # --- Step 7-8: Heuristic component for put_on_tray actions ---
        # Count sandwiches that are already on trays
        num_ontray_sandwiches = len(ontray_sandwiches_set)
        # Number of sandwiches that still need to be put on a tray from the kitchen
        h_put_on_tray = max(0, num_unserved_total - num_ontray_sandwiches)

        # --- Step 9-12: Heuristic component for move actions ---
        # Places where unserved children are waiting
        waiting_places_of_unserved = {self.child_places[c] for c in unserved_children}
        # Places that currently have trays
        places_with_trays = set(trays_at_places_state.keys())
        # Places needing a tray where unserved children are waiting
        places_needing_tray = waiting_places_of_unserved - places_with_trays
        # Minimum moves needed is the number of places that need a tray
        h_move = len(places_needing_tray)

        # --- Step 13: Total heuristic ---
        # Summing up the required steps from different stages.
        total_heuristic = h_serve + h_put_on_tray + h_make + h_move

        return total_heuristic
