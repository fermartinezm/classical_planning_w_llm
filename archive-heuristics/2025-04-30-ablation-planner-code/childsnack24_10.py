from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    This heuristic estimates the number of actions needed to serve all children
    by summing up the estimated costs for each unserved child independently.
    The cost for a child depends on how far along the process (make, transport, serve)
    their required snack is.

    # Assumptions
    - Each child needs exactly one sandwich.
    - Resources (ingredients, sandwich objects, trays) are assumed to be eventually available
      when needed for the *first* action in a sequence (make, put, move). Resource
      contention and limited quantities are not strictly modeled beyond checking
      if *any* suitable sandwich exists.
    - The heuristic counts the minimum number of actions (make, put_on_tray, move_tray, serve)
      required for each child, assuming these actions can be performed if the
      preceding steps are met.

    # Heuristic Initialization
    - Extracts the set of children that need to be served from the goal conditions.
    - Extracts static information about children: allergy status and waiting places.
    - Extracts all sandwich and tray object names from the problem definition.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic calculates the cost for each unserved child independently and sums these costs.
    For an unserved child C waiting at place P:

    1.  **Base Cost (Serve):** Add 1 action for the final 'serve' action.

    2.  **Transport/Put Cost:** Check if a suitable sandwich S is already on a tray T that is currently at the child's waiting place P.
        - A sandwich S is "suitable" for child C if it is gluten-free (predicate `no_gluten_sandwich`) when C is allergic (`allergic_gluten`), or any sandwich otherwise.
        - If NO suitable sandwich is found on a tray at P: Add 1 action. This represents the cost of either moving a tray to P (if a suitable sandwich is on a tray elsewhere) or putting a sandwich onto a tray and potentially moving it (if the sandwich is in the kitchen).

    3.  **Make Cost:** If the previous check failed (no suitable sandwich on a tray at P), check if *any* suitable sandwich S exists anywhere (either `at_kitchen_sandwich` or `ontray` any tray).
        - If NO suitable sandwich exists anywhere: Add 1 action. This represents the cost of making the sandwich (`make_sandwich` or `make_sandwich_no_gluten`).

    The total heuristic value is the sum of these calculated costs (0, 1, 2, or 3) for every child that is not yet served according to the goal.
    - Cost 0: Child is served.
    - Cost 1: Suitable sandwich is on a tray at the child's location (needs only serve).
    - Cost 2: Suitable sandwich exists but is not on a tray at the child's location (needs transport/put + serve).
    - Cost 3: Suitable sandwich does not exist anywhere (needs make + transport/put + serve).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and object types.
        """
        # Extract children that need to be served from the goal conditions.
        self.goal_served_children = {
            get_parts(goal)[1]
            for goal in task.goals
            if match(goal, "served", "*")
        }

        # Extract static information about children and places.
        self.child_allergy = {}
        self.child_waiting_place = {}

        # Extract object names by type from the task definition
        # The task object is assumed to have an 'objects' attribute which is a list of Object instances
        # where each Object instance has 'name' and 'type' attributes.
        # This structure is implied by how the example heuristics access task.objects.
        self.all_children = {obj.name for obj in task.objects if obj.type == 'child'}
        self.all_trays = {obj.name for obj in task.objects if obj.type == 'tray'}
        self.all_sandwiches = {obj.name for obj in task.objects if obj.type == 'sandwich'}
        self.all_places = {obj.name for obj in task.objects if obj.type == 'place'}
        # Add kitchen constant if it's not explicitly in objects (PDDL constants are usually listed)
        # Assuming 'kitchen' is always a place and a constant.
        self.all_places.add('kitchen')


        # Extract facts from task.static (frozenset of strings)
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "allergic_gluten":
                child_name = parts[1]
                if child_name in self.all_children: # Ensure it's a known child object
                    self.child_allergy[child_name] = True
            elif predicate == "not_allergic_gluten":
                 child_name = parts[1]
                 if child_name in self.all_children: # Ensure it's a known child object
                    self.child_allergy[child_name] = False # Store explicitly False
            elif predicate == "waiting":
                child_name = parts[1]
                place_name = parts[2]
                if child_name in self.all_children and place_name in self.all_places: # Ensure known objects
                    self.child_waiting_place[child_name] = place_name

        # Note: It's assumed that for every child in self.goal_served_children,
        # there will be a corresponding 'waiting' fact in the initial state (or static facts).
        # If a child in the goal is not listed as 'waiting', the lookup
        # self.child_waiting_place.get(child) will return None, and the heuristic
        # calculation for that child will proceed assuming no suitable sandwich
        # is at their (unknown) location, leading to a higher cost (likely 2 or 3).
        # This is a reasonable fallback for potentially malformed problems.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state as a frozenset of strings.

        # Extract relevant state information
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        at_kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        ontray_sandwiches = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "ontray", "*", "*")}
        no_gluten_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        # We need tray locations for trays that have sandwiches on them or might be used.
        # It's simpler to just get all tray locations that are currently asserted.
        tray_locations = {}
        for fact in state:
             parts = get_parts(fact)
             if match(fact, "at", "*", "*") and len(parts) == 3 and parts[1] in self.all_trays and parts[2] in self.all_places:
                 tray_locations[parts[1]] = parts[2]

        total_cost = 0

        # Calculate cost for each unserved child
        for child in self.goal_served_children:
            if child in served_children:
                continue # Child is already served, cost is 0 for this child

            # Child is unserved, calculate cost
            child_cost = 0

            # 1. Cost for the final 'serve' action
            child_cost += 1

            # Get child's waiting place and allergy status
            waiting_place = self.child_waiting_place.get(child)
            # If waiting_place is None, the child cannot be served at a known location.
            # This state is likely unsolvable for this child.
            # The heuristic will assign max cost (3) in this case, which is acceptable.

            is_allergic = self.child_allergy.get(child, False) # Default to not allergic if status missing

            # 2. Check if a suitable sandwich is already on a tray at the child's location
            suitable_on_tray_at_place = False
            if waiting_place: # Only check if waiting place is known
                for s, t in ontray_sandwiches:
                    # Check if sandwich s is suitable for child
                    s_is_gluten_free = s in no_gluten_sandwiches
                    is_suitable = s_is_gluten_free if is_allergic else True # Any sandwich is suitable if not allergic

                    # Check if the tray t is at the child's waiting place
                    t_at_place = t in tray_locations and tray_locations[t] == waiting_place

                    if is_suitable and t_at_place:
                        suitable_on_tray_at_place = True
                        break # Found one, no need to check further for this child's stage

            if not suitable_on_tray_at_place:
                # Need to get sandwich/tray to location (move or put)
                child_cost += 1

                # 3. Check if a suitable sandwich exists anywhere (kitchen or on any tray)
                suitable_sandwich_exists_anywhere = False

                # Check suitable sandwiches in the kitchen
                for s in at_kitchen_sandwiches:
                    s_is_gluten_free = s in no_gluten_sandwiches
                    is_suitable = s_is_gluten_free if is_allergic else True
                    if is_suitable:
                        suitable_sandwich_exists_anywhere = True
                        break # Found one in kitchen

                # If not found in kitchen, check suitable sandwiches on any tray
                if not suitable_sandwich_exists_anywhere:
                    for s, t in ontray_sandwiches:
                        s_is_gluten_free = s in no_gluten_sandwiches
                        is_suitable = s_is_gluten_free if is_allergic else True
                        if is_suitable:
                            suitable_sandwich_exists_anywhere = True
                            break # Found one on a tray

                if not suitable_sandwich_exists_anywhere:
                    # Need to make the sandwich
                    child_cost += 1

            total_cost += child_cost

        return total_cost
