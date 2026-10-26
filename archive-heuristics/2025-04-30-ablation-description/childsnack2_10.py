# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch # Keep import for completeness, although not strictly used by get_parts/match

# Mock Heuristic base class if running standalone or for testing purposes
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # This should not happen with valid PDDL facts represented as strings
        # print(f"Warning: Unexpected fact format: {fact}")
        return []

    # Handle potential empty facts like "()" if they could occur
    content = fact[1:-1].strip()
    if not content:
        return []

    return content.split()


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve
    each unserved child independently, based on the current state of suitable
    sandwiches and ingredients. The total heuristic value is the sum of these
    individual minimum costs.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - The cost for a child is determined by the most advanced stage a suitable
      sandwich has reached relative to that child:
        - Stage 4: Suitable sandwich on tray at child's location (cost 1: serve).
        - Stage 3: Suitable sandwich on tray at kitchen (cost 2: move_tray, serve).
        - Stage 2: Suitable sandwich at kitchen, not on tray (cost 3: put_on_tray, move_tray, serve).
        - Stage 1: Ingredients available to make a suitable sandwich (cost 4: make, put_on_tray, move_tray, serve).
        - Stage 0: Ingredients not available (cost infinity).
    - This heuristic sums the minimum costs per child, ignoring potential
      resource sharing (e.g., one tray move serving multiple children at the
      same location, or shared ingredients/sandwiches beyond the minimum needed
      to make *one* suitable sandwich for a child). This makes the heuristic admissible.

    # Heuristic Initialization
    The constructor extracts static information from the task:
    - Which children are allergic to gluten.
    - Which place each child is waiting at.
    - Which bread portions are no-gluten.
    - Which content portions are no-gluten.
    - Identifies all children involved in the problem (from allergy/waiting facts).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Parse the current state to identify: served children, items/sandwiches
       at the kitchen, sandwich gluten status, tray locations, and sandwiches
       currently on trays.
    3. Identify all children who are not yet served by comparing the set of
       all children (from static facts) with the set of served children (from state).
    4. For each unserved child:
       a. Determine the child's waiting place and allergy status using the
          pre-computed static information.
       b. Determine the type of sandwich required (no-gluten if allergic, any if not).
       c. Check if a suitable sandwich is currently on a tray located at the
          child's waiting place. If yes, the minimum cost for this child is 1
          (for the 'serve' action).
       d. If not, check if a suitable sandwich is currently on a tray located
          at the kitchen. If yes, the minimum cost for this child is 2
          (for 'move_tray' and 'serve').
       e. If not, check if a suitable sandwich is currently at the kitchen
          (but not on a tray). If yes, the minimum cost for this child is 3
          (for 'put_on_tray', 'move_tray', and 'serve').
       f. If not, check if the necessary ingredients (bread and content) are
          available at the kitchen to make a suitable sandwich of the required
          type. If yes, the minimum cost for this child is 4
          (for 'make_sandwich', 'put_on_tray', 'move_tray', and 'serve').
       g. If the necessary ingredients are not available, the problem is
          unsolvable from this state for this child, and the heuristic returns
          infinity immediately.
       h. Add the determined minimum cost for this child to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.

        @param task: The planning task object containing initial state, goals, etc.
        """
        super().__init__(task)

        # Extract static information from task.static
        self.child_allergy = {}  # child_name -> True if allergic, False otherwise
        self.child_place = {}    # child_name -> waiting_place
        self.bread_gluten = set()   # Set of no_gluten bread names
        self.content_gluten = set() # Set of no_gluten content names
        self.all_children = set() # Set of all child names involved in the problem

        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            if predicate == 'allergic_gluten':
                child_name = parts[1]
                self.child_allergy[child_name] = True
                self.all_children.add(child_name)
            elif predicate == 'not_allergic_gluten':
                child_name = parts[1]
                self.child_allergy[child_name] = False
                self.all_children.add(child_name)
            elif predicate == 'waiting':
                child_name = parts[1]
                place_name = parts[2]
                self.child_place[child_name] = place_name
                self.all_children.add(child_name) # Ensure children mentioned in waiting are added
            elif predicate == 'no_gluten_bread':
                self.bread_gluten.add(parts[1])
            elif predicate == 'no_gluten_content':
                self.content_gluten.add(parts[1])

        # Ensure all children mentioned in goals are also in our list (they should be via waiting/allergy)
        # This is a safety check, typically children in goals are also in init (waiting/allergy)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == 'served':
                 self.all_children.add(parts[1])


    def _is_suitable_sandwich(self, sandwich_name, child_name, state_data):
        """
        Checks if a given sandwich is suitable for a given child based on allergy.

        @param sandwich_name: The name of the sandwich object.
        @param child_name: The name of the child object.
        @param state_data: Dictionary containing parsed state facts.
        @return: True if the sandwich is suitable, False otherwise.
        """
        # Get allergy status from pre-computed static data
        child_allergic = self.child_allergy.get(child_name, False) # Default to not allergic if not specified

        # Check if the sandwich is a no-gluten sandwich based on state data
        sandwich_is_ng = sandwich_name in state_data['no_gluten_sandwich']

        if child_allergic:
            # Allergic children *must* have a no-gluten sandwich
            return sandwich_is_ng
        else:
            # Non-allergic children can have any sandwich (regular or no-gluten)
            return True

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.

        @param node: The search node containing the current state.
        @return: The estimated cost to reach a goal state.
        """
        state = node.state

        # Parse relevant state facts into a dictionary for easier access
        state_data = {
            'served_children': set(),
            'at_kitchen_bread': set(),
            'at_kitchen_content': set(),
            'at_kitchen_sandwich': set(),
            'no_gluten_sandwich': set(),
            'tray_location': {}, # tray_name -> place_name
            'sandwich_on_tray': {}, # sandwich_name -> tray_name
            # 'notexist_sandwiches': set(), # Not strictly needed for this heuristic logic
        }

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            if predicate == 'served':
                state_data['served_children'].add(parts[1])
            elif predicate == 'at_kitchen_bread':
                state_data['at_kitchen_bread'].add(parts[1])
            elif predicate == 'at_kitchen_content':
                state_data['at_kitchen_content'].add(parts[1])
            elif predicate == 'at_kitchen_sandwich':
                state_data['at_kitchen_sandwich'].add(parts[1])
            elif predicate == 'no_gluten_sandwich':
                 state_data['no_gluten_sandwich'].add(parts[1])
            elif predicate == 'at':
                # This predicate is used for trays and potentially other objects/constants
                obj_name = parts[1]
                place_name = parts[2]
                # We only care about tray locations for the heuristic
                # Assuming objects starting with 'tray' are trays based on domain types
                if obj_name.startswith('tray'):
                     state_data['tray_location'][obj_name] = place_name
            elif predicate == 'ontray':
                sandwich_name = parts[1]
                tray_name = parts[2]
                state_data['sandwich_on_tray'][sandwich_name] = tray_name
            # 'notexist' facts are not directly used in this heuristic's cost calculation


        total_cost = 0
        # Identify children who are in the set of all children but not in the served set
        unserved_children = [c for c in self.all_children if c not in state_data['served_children']]

        # If there are no unserved children, the goal is reached
        if not unserved_children:
            return 0

        # Calculate cost for each unserved child independently
        for child_name in unserved_children:
            # Get static information for the child
            child_place = self.child_place.get(child_name)
            child_allergic = self.child_allergy.get(child_name, False)

            # If a child doesn't have a waiting place defined in static facts,
            # the problem instance might be malformed or unsolvable for this child.
            # Return infinity in such cases.
            if child_place is None:
                 # print(f"Error: Child {child_name} has no waiting place defined in static facts.")
                 return float('inf')

            cost_for_child = 0
            sandwich_found_at_stage = False # Flag to stop checking lower stages once a stage is met

            # Check Stage 4: Suitable sandwich on tray at child's place? (Cost 1: serve)
            # Iterate through sandwiches known to be on trays
            for s_name, t_name in state_data['sandwich_on_tray'].items():
                # Check if the tray is at the child's location
                if state_data['tray_location'].get(t_name) == child_place:
                    # Check if the sandwich is suitable for the child
                    if self._is_suitable_sandwich(s_name, child_name, state_data):
                        cost_for_child = 1
                        sandwich_found_at_stage = True
                        break # Found a sandwich at the highest stage, no need to check lower stages for this child

            # If not found at Stage 4, check Stage 3
            if not sandwich_found_at_stage:
                # Check Stage 3: Suitable sandwich on tray at kitchen? (Cost 2: move_tray, serve)
                for s_name, t_name in state_data['sandwich_on_tray'].items():
                     # Check if the tray is at the kitchen
                     if state_data['tray_location'].get(t_name) == 'kitchen':
                         # Check if the sandwich is suitable for the child
                         if self._is_suitable_sandwich(s_name, child_name, state_data):
                             cost_for_child = 2
                             sandwich_found_at_stage = True
                             break # Found a sandwich at this stage

            # If not found at Stage 3, check Stage 2
            if not sandwich_found_at_stage:
                # Check Stage 2: Suitable sandwich at kitchen (not on tray)? (Cost 3: put_on_tray, move_tray, serve)
                # Iterate through sandwiches known to be at the kitchen (not on trays)
                for s_name in state_data['at_kitchen_sandwich']:
                     # Check if the sandwich is suitable for the child
                     if self._is_suitable_sandwich(s_name, child_name, state_data):
                         cost_for_child = 3
                         sandwich_found_at_stage = True
                         break # Found a sandwich at this stage

            # If not found at Stage 2, check Stage 1
            if not sandwich_found_at_stage:
                # Check Stage 1: Ingredients exist to make suitable sandwich? (Cost 4: make, put_on_tray, move_tray, serve)
                ingredients_exist = False
                if child_allergic:
                    # Allergic child needs NG bread and NG content at kitchen
                    # Check if there is at least one NG bread at kitchen and one NG content at kitchen
                    has_ng_bread_at_kitchen = any(b in self.bread_gluten for b in state_data['at_kitchen_bread'])
                    has_ng_content_at_kitchen = any(c_ in self.content_gluten for c_ in state_data['at_kitchen_content'])
                    if has_ng_bread_at_kitchen and has_ng_content_at_kitchen:
                        ingredients_exist = True
                else:
                    # Non-allergic child needs any bread and any content at kitchen
                    if state_data['at_kitchen_bread'] and state_data['at_kitchen_content']:
                        ingredients_exist = True

                if ingredients_exist:
                    cost_for_child = 4
                    sandwich_found_at_stage = True # Ingredients exist to reach this stage
                else:
                    # Stage 0: Cannot make suitable sandwich, unsolvable for this child
                    # Return infinity immediately as the problem is unsolvable
                    return float('inf')

            # Add the determined minimum cost for this child to the total
            total_cost += cost_for_child

        # Return the sum of minimum costs for all unserved children
        return total_cost
