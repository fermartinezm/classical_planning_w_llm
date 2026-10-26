# Import necessary modules
from heuristics.heuristic_base import Heuristic
from task import Task # Assuming Task class is available in the 'task' module
import math # For float('inf')

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    This heuristic estimates the number of actions required to reach a goal state
    by summing up the estimated costs for serving each unserved child. It breaks
    down the process into stages: serving the child, delivering a sandwich on a
    tray to their location, putting a sandwich onto a tray, and making a sandwich.
    It counts the number of actions needed for each stage based on the current
    availability of suitable sandwiches in different locations (on trays at the
    correct place, on trays elsewhere, in the kitchen, or needing to be made)
    and the needs of the unserved children (allergy status).

    Assumptions:
    - Unit cost for all actions.
    - Trays are initially in the kitchen or can be moved there.
    - Ingredients and sandwich objects are consumed when making sandwiches.
    - The heuristic provides an estimate and is not guaranteed to be admissible,
      prioritizing reducing expanded nodes in a greedy best-first search.
    - The domain provides sufficient resources (bread, content, sandwich objects,
      trays) to solve the problem if it is solvable; if not, the heuristic returns infinity.
    - Children's waiting places and allergy statuses are static.

    Heuristic Initialization:
    In the constructor, the heuristic parses the static facts from the task
    description to pre-compute and store information that does not change
    during the search. This includes:
    - Identifying which children are allergic or not allergic.
    - Mapping each child to their waiting place.
    - Identifying which bread and content items are gluten-free.
    It also identifies the set of children that need to be served to reach the goal.

    Step-By-Step Thinking for Computing Heuristic:
    1.  **Parse State:** The current state facts are parsed to determine:
        - Which children are already served.
        - Which sandwiches are on which trays (`ontray_map`).
        - Where each tray is located (`tray_location_map`).
        - Which sandwiches, bread, and content are currently in the kitchen.
        - Which sandwich objects do not yet exist (`notexist`).
        - Which sandwiches are gluten-free in the current state (`no_gluten_sandwich`).
    2.  **Identify Unserved Children:** Determine the set of children from the goal
        who are not yet served in the current state. If this set is empty, the
        goal is reached, and the heuristic value is 0.
    3.  **Base Cost (Serve Action):** The heuristic starts with a cost equal to
        the number of unserved children. This represents the minimum number of
        `serve_sandwich` actions required.
    4.  **Determine Allergy Status of Unserved:** Check if all unserved children
        are allergic. This determines the type of sandwich (gluten-free or any)
        required for the remaining steps.
    5.  **Cost for Delivery (Move Tray Action):** For each unserved child, check
        if there is already a suitable sandwich on a tray located at their waiting
        place. Count the number of unserved children for whom this condition is
        NOT met. Each such child requires a sandwich to be delivered, which
        involves at least one `move_tray` action (assuming a tray with a suitable
        sandwich becomes available elsewhere). Add this count to the heuristic.
    6.  **Cost for Putting on Tray (Put on Tray Action):** Count the number of
        suitable sandwiches that are currently in the kitchen (`at_kitchen_sandwich`).
        These sandwiches need a `put_on_tray` action. Add this count to the heuristic.
    7.  **Count Available Suitable Sandwiches:** Determine the total number of
        suitable sandwiches that currently exist, either on trays (anywhere) or
        in the kitchen.
    8.  **Cost for Making Sandwiches (Make Sandwich Action):** Calculate how many
        *additional* suitable sandwiches are needed by subtracting the number of
        available suitable sandwiches (from step 7) from the total number of
        unserved children. If this number is positive, these sandwiches must be
        made. Add this count to the heuristic (cost of `make_sandwich`).
    9.  **Cost for Putting Newly Made Sandwiches on Tray:** The sandwiches counted
        in step 8, once made, will be in the kitchen and will also need a
        `put_on_tray` action. Add the count from step 8 again to the heuristic.
    10. **Check Solvability:** If the number of sandwiches needing to be made
        (from step 8) exceeds the available ingredients (gluten-free ingredients
        if all unserved are allergic, or any ingredients otherwise) and available
        `notexist` sandwich objects in the kitchen, the state is likely unsolvable
        within the domain's actions, and the heuristic returns infinity.
    11. **Return Total Cost:** The sum of costs from steps 3, 5, 6, 8, and 9 is
        returned as the heuristic value.
    """

    def __init__(self, task: Task):
        """
        Initializes the heuristic by parsing static task information.

        Args:
            task: The planning task object.
        """
        super().__init__()
        self.task = task # Store task for access to goals later
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.child_waiting_place = {} # child -> place
        self.no_gluten_bread_items = set()
        self.no_gluten_content_items = set()

        # Parse static facts from the task
        for fact_string in task.static:
            predicate, args = self.parse_fact(fact_string)
            if predicate == 'allergic_gluten':
                if len(args) == 1:
                    self.allergic_children.add(args[0])
            elif predicate == 'not_allergic_gluten':
                if len(args) == 1:
                    self.not_allergic_children.add(args[0])
            elif predicate == 'waiting':
                if len(args) == 2: # Ensure correct number of arguments
                    self.child_waiting_place[args[0]] = args[1]
            elif predicate == 'no_gluten_bread':
                if len(args) == 1:
                    self.no_gluten_bread_items.add(args[0])
            elif predicate == 'no_gluten_content':
                if len(args) == 1:
                    self.no_gluten_content_items.add(args[0])

        # Get goal children from the task goals
        self.goal_children = set()
        for fact_string in task.goals:
            predicate, args = self.parse_fact(fact_string)
            if predicate == 'served':
                if len(args) == 1:
                    self.goal_children.add(args[0])

    def parse_fact(self, fact_string):
        """Helper to parse a PDDL fact string into predicate and arguments."""
        # Remove surrounding brackets and split by space
        # Handle potential empty args or complex structures if necessary,
        # but based on domain, simple split should work.
        parts = fact_string[1:-1].split()
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def is_suitable(self, sandwich, child, allergic_children_set, no_gluten_sandwiches_state_set):
        """Checks if a sandwich is suitable for a specific child."""
        is_child_allergic = child in allergic_children_set

        if is_child_allergic:
            # Allergic children require gluten-free sandwiches
            return sandwich in no_gluten_sandwiches_state_set
        else:
            # Non-allergic children can have any sandwich
            return True

    def is_suitable_for_any_unserved(self, sandwich, all_unserved_are_allergic, no_gluten_sandwiches_state_set):
        """Checks if a sandwich is suitable for at least one unserved child."""
        if all_unserved_are_allergic:
            # If all unserved children are allergic, the sandwich must be gluten-free
            return sandwich in no_gluten_sandwiches_state_set
        else:
            # If there's at least one non-allergic unserved child, any sandwich is suitable
            return True

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.

        Args:
            node: The search node containing the state.

        Returns:
            The estimated number of actions to reach a goal state, or infinity
            if the state is estimated to be unsolvable.
        """
        state = node.state

        # Parse state facts into temporary data structures
        served_children = set()
        ontray_map = {} # sandwich -> tray
        tray_location_map = {} # tray -> place
        kitchen_sandwiches = set()
        notexist_sandwiches = set()
        kitchen_bread = set()
        kitchen_content = set()
        no_gluten_sandwiches_in_state = set()

        for fact_string in state:
            predicate, args = self.parse_fact(fact_string)
            if predicate == 'served':
                if len(args) == 1:
                    served_children.add(args[0])
            elif predicate == 'ontray':
                if len(args) == 2:
                    ontray_map[args[0]] = args[1]
            elif predicate == 'at':
                if len(args) == 2:
                    tray_location_map[args[0]] = args[1]
            elif predicate == 'at_kitchen_sandwich':
                if len(args) == 1:
                    kitchen_sandwiches.add(args[0])
            elif predicate == 'notexist':
                if len(args) == 1:
                    notexist_sandwiches.add(args[0])
            elif predicate == 'at_kitchen_bread':
                if len(args) == 1:
                    kitchen_bread.add(args[0])
            elif predicate == 'at_kitchen_content':
                if len(args) == 1:
                    kitchen_content.add(args[0])
            elif predicate == 'no_gluten_sandwich':
                if len(args) == 1:
                    no_gluten_sandwiches_in_state.add(args[0])

        # Identify unserved children
        unserved_children = self.goal_children - served_children

        # If all goal children are served, the heuristic is 0
        if not unserved_children:
            return 0

        # --- Heuristic Calculation ---
        h = len(unserved_children) # Base cost: 1 action (serve) per unserved child

        # Determine if all unserved children are allergic (affects sandwich suitability)
        all_unserved_are_allergic = all(c in self.allergic_children for c in unserved_children)

        # Cost for Delivery (Move Tray Action)
        # Count children who don't have a suitable sandwich already at their location on a tray
        needs_delivery_count = 0
        for child in unserved_children:
            child_place = self.child_waiting_place.get(child)
            # If a child is unserved but not waiting anywhere (shouldn't happen based on domain), it's unsolvable
            if child_place is None:
                 return float('inf')

            found_suitable_at_location = False
            # Check sandwiches currently on trays
            for sandwich, tray in ontray_map.items():
                # Check if the tray is at the child's location
                if tray in tray_location_map and tray_location_map[tray] == child_place:
                    # Check if the sandwich is suitable for this specific child
                    if self.is_suitable(sandwich, child, self.allergic_children, no_gluten_sandwiches_in_state):
                        found_suitable_at_location = True
                        break # Found a suitable sandwich at the location for this child

            if not found_suitable_at_location:
                needs_delivery_count += 1 # This child needs a sandwich delivered (implies a move_tray)

        h += needs_delivery_count # Add cost for move_tray actions

        # Cost for Putting on Tray (Put on Tray Action) for sandwiches already in kitchen
        n_in_kitchen_suitable_count = 0
        for sandwich in kitchen_sandwiches:
            # Check if the sandwich is suitable for *any* unserved child
            if self.is_suitable_for_any_unserved(sandwich, all_unserved_are_allergic, no_gluten_sandwiches_in_state):
                n_in_kitchen_suitable_count += 1

        h += n_in_kitchen_suitable_count # Add cost for put_on_tray actions for kitchen sandwiches

        # Calculate Num_Sandwiches_To_Make
        # Count suitable sandwiches that currently exist (on trays or in kitchen)
        available_suitable_sandwiches_count = 0
        counted_sandwiches = set() # Use a set to avoid double counting if a sandwich is somehow listed twice (e.g., ontray and in kitchen - though domain prevents this)

        # Count suitable sandwiches on trays
        for sandwich in ontray_map:
             # Check if the sandwich is suitable for *any* unserved child
            if self.is_suitable_for_any_unserved(sandwich, all_unserved_are_allergic, no_gluten_sandwiches_in_state):
                available_suitable_sandwiches_count += 1
                counted_sandwiches.add(sandwich)

        # Count suitable sandwiches in kitchen (only those not already counted, though unlikely)
        for sandwich in kitchen_sandwiches:
             # Check if the sandwich is suitable for *any* unserved child
            if sandwich not in counted_sandwiches and self.is_suitable_for_any_unserved(sandwich, all_unserved_are_allergic, no_gluten_sandwiches_in_state):
                available_suitable_sandwiches_count += 1
                counted_sandwiches.add(sandwich)

        # Number of sandwiches that still need to be made
        num_sandwiches_to_make = max(0, len(unserved_children) - available_suitable_sandwiches_count)

        # Check if enough ingredients and sandwich objects exist to make the required sandwiches
        if num_sandwiches_to_make > 0:
            n_notexist = len(notexist_sandwiches)
            if all_unserved_are_allergic:
                # Need gluten-free ingredients
                n_gf_bread = len([b for b in kitchen_bread if b in self.no_gluten_bread_items])
                n_gf_content = len([c for c in kitchen_content if c in self.no_gluten_content_items])
                max_makeable = min(n_gf_bread, n_gf_content, n_notexist)
            else:
                # Any ingredients are fine
                n_any_bread = len(kitchen_bread)
                n_any_content = len(kitchen_content)
                max_makeable = min(n_any_bread, n_any_content, n_notexist)

            # If we need to make more sandwiches than is possible, it's unsolvable
            if num_sandwiches_to_make > max_makeable:
                return float('inf')

        # Add cost for making sandwiches (make_sandwich action)
        h += num_sandwiches_to_make

        # Add cost for putting newly made sandwiches on a tray (put_on_tray action)
        h += num_sandwiches_to_make

        return h
