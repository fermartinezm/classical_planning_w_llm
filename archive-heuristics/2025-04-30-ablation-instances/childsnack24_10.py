from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not isinstance(fact, str) or not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It calculates the minimum steps needed to get a suitable sandwich to each unserved child's location and adds the final 'serve' action. The total heuristic is the sum of these costs for all unserved children.

    # Assumptions
    - All children specified in the static facts need to be served to reach a goal state.
    - Children remain waiting at their initial locations throughout the plan.
    - Ingredients (bread, content) and available sandwich names are sufficient to make new sandwiches if needed (a simplifying relaxation for efficiency).
    - Trays are always available and can be moved.
    - The cost of each action (make, put, move, serve) is 1.

    # Heuristic Initialization
    - Identify all children and their waiting locations from the static facts.
    - Identify which children are allergic to gluten from the static facts.
    - Store the set of all children that need to be served (all children mentioned in static facts).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize the total heuristic cost to 0.
    2. Identify the set of children who are currently `served` in the given state.
    3. Iterate through all children identified during the heuristic's initialization (all children mentioned in the static facts).
    4. For each child:
        a. Check if the child is in the set of `served` children. If yes, this child contributes 0 to the heuristic, so continue to the next child.
        b. If the child is not `served`:
            i. Retrieve the child's waiting location `?p` and allergy status (allergic or not) from the information extracted during initialization.
            ii. Determine what kind of sandwich is *suitable* for this child: if the child is allergic, only a `no_gluten_sandwich` is suitable; otherwise, any sandwich is suitable.
            iii. Calculate the minimum number of actions required to get a suitable sandwich onto a tray and move that tray to the child's location `?p`. This is the 'delivery cost'.
                - Initialize `min_delivery_cost = 3`. This represents the cost of making a *new* suitable sandwich (1 action), putting it on a tray (1 action), and moving that tray to the child's location (1 action). This assumes, for simplicity, that ingredients and a tray at the kitchen are always available if needed.
                - Examine the current state to find existing sandwiches and their locations:
                    - If a suitable sandwich `?s` is found that is currently `(ontray ?s ?t)`:
                        - Find the current location of tray `?t` by looking for `(at ?t ?p_current)` in the state.
                        - If `?p_current` is the same as the child's waiting location `?p`, the sandwich is already where it needs to be for serving. Update `min_delivery_cost = min(min_delivery_cost, 0)`.
                        - If `?p_current` is different from `?p`, the tray needs to be moved. Update `min_delivery_cost = min(min_delivery_cost, 1)` (cost of `move_tray`).
                    - If a suitable sandwich `?s` is found that is currently `(at_kitchen_sandwich ?s)`:
                        - This sandwich needs to be put on a tray (`put_on_tray`, 1 action) and then the tray needs to be moved to the child's location (`move_tray`, 1 action). Update `min_delivery_cost = min(min_delivery_cost, 2)`. (This assumes a tray is available at the kitchen).
                - After checking all existing suitable sandwiches, `min_delivery_cost` holds the minimum estimated actions to get a suitable sandwich to the child's location.
            iv. The total estimated cost for this unserved child is `min_delivery_cost` + 1 (for the final `serve_sandwich` action). Add this cost to the `total_heuristic_cost`.
    5. Return the `total_heuristic_cost`. This value is 0 if and only if all children are served (the goal state).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children,
        their waiting locations, and allergy status from the task's static facts.
        """
        self.goals = task.goals # Goal conditions (primarily used to identify goal state, but static facts define children)
        static_facts = task.static # Facts that are not affected by actions.

        # Extract children, their waiting places, and allergy status from static facts
        self.children_info = {} # {child_name: {'place': place_name, 'allergic': bool}}
        self.all_children = set() # Set of all children mentioned in static facts

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "waiting" and len(parts) == 3:
                child, place = parts[1], parts[2]
                if child not in self.children_info:
                    self.children_info[child] = {'place': place, 'allergic': False} # Default allergic to False
                else:
                     self.children_info[child]['place'] = place
                self.all_children.add(child)

            elif predicate == "allergic_gluten" and len(parts) == 2:
                child = parts[1]
                if child not in self.children_info:
                     self.children_info[child] = {'place': None, 'allergic': True} # Place will be filled by 'waiting' fact
                else:
                     self.children_info[child]['allergic'] = True
                self.all_children.add(child)

            elif predicate == "not_allergic_gluten" and len(parts) == 2:
                 child = parts[1]
                 if child not in self.children_info:
                     self.children_info[child] = {'place': None, 'allergic': False} # Place will be filled by 'waiting' fact
                 else:
                     self.children_info[child]['allergic'] = False
                 self.all_children.add(child)

        # Note: Information about no_gluten_bread/content is static, but not strictly needed
        # for this relaxed heuristic that assumes ingredients are available if needed.
        # Information about no_gluten_sandwich is dynamic and checked in __call__.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        total_heuristic_cost = 0

        # Extract dynamic information about the state
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        sandwich_locations = {} # {sandwich_name: location_or_tray_name}
        tray_locations = {} # {tray_name: place_name}
        gluten_free_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        # Populate dynamic location information
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at_kitchen_sandwich" and len(parts) == 2:
                sandwich = parts[1]
                sandwich_locations[sandwich] = "kitchen"
            elif predicate == "ontray" and len(parts) == 3:
                sandwich, tray = parts[1], parts[2]
                sandwich_locations[sandwich] = tray # Store the tray name as the location
            elif predicate == "at" and len(parts) == 3:
                 tray, place = parts[1], parts[2]
                 tray_locations[tray] = place

        # Iterate through all children who are initially waiting (from static facts)
        for child in self.all_children:
            # If the child is already served in the current state, they don't contribute to the heuristic
            if child in served_children:
                continue

            # Child is not served, calculate the estimated cost for this child
            child_info = self.children_info.get(child)
            if not child_info:
                 # Should not happen if all_children comes from static facts containing waiting/allergy
                 continue

            child_place = child_info['place']
            child_allergic = child_info['allergic']

            # Minimum cost to get a suitable sandwich to child_place on a tray
            # Base cost assumes making a new sandwich is needed:
            # make (1) + put_on_tray (1) + move_tray (1) = 3 actions
            min_delivery_cost = 3

            # Check existing sandwiches to see if a lower delivery cost is possible
            for sandwich, current_loc in sandwich_locations.items():
                # Determine if the current sandwich is suitable for this child
                is_suitable = True
                if child_allergic and sandwich not in gluten_free_sandwiches:
                    is_suitable = False

                if is_suitable:
                    if current_loc == "kitchen":
                        # Sandwich is in the kitchen.
                        # Needs put_on_tray (1) + move_tray to child's place (1) = 2 actions
                        min_delivery_cost = min(min_delivery_cost, 2)
                    elif current_loc in tray_locations: # current_loc is a tray name
                        tray = current_loc
                        tray_place = tray_locations.get(tray) # Get the physical location of the tray

                        if tray_place == child_place:
                            # Sandwich is on a tray that is already at the child's location.
                            # Delivery cost is 0.
                            min_delivery_cost = min(min_delivery_cost, 0)
                        elif tray_place is not None: # Tray is somewhere else
                            # Sandwich is on a tray elsewhere.
                            # Needs move_tray from current tray_place to child_place (1 action).
                            min_delivery_cost = min(min_delivery_cost, 1)
                        # If tray_place is None, the state is inconsistent (ontray but tray has no location).
                        # We ignore this case assuming valid states.

            # The total cost for this child is the minimum delivery cost found + the serve action (1)
            total_heuristic_cost += min_delivery_cost + 1

        # The total heuristic value is the sum of costs for all unserved children.
        # It is 0 if and only if all children are served.
        return total_heuristic_cost
