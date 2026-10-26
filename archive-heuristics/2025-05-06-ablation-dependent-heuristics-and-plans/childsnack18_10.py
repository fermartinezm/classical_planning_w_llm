from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments, unless args has a wildcard at the end
    if len(args) > len(parts) and args[-1] != "*":
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It calculates a cost for each unserved child based on the "distance" of a suitable
    sandwich from being served to that child. The cost is higher if the sandwich
    needs to be made from scratch compared to being already on a tray at the child's location.

    # Assumptions
    - Each unserved child needs one suitable sandwich.
    - The cost for each child can be estimated independently and summed up.
    - Resources (trays, components, sandwich slots) are considered in a simplified way
      when determining the "closest" state of a suitable sandwich. Resource contention
      is not explicitly modeled, making the heuristic non-admissible but potentially
      effective for greedy search.
    - Components (bread, content) are assumed sufficient if at least one of the required
      type is at the kitchen when checking if a sandwich can be made.
    - A tray is assumed available at the kitchen when needed for 'put_on_tray'.

    # Heuristic Initialization
    - Extracts static information about which children are allergic and which
      bread/content portions are no-gluten.
    - Identifies all children objects from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Identify all children in the problem.
    2. Identify which of these children are currently unserved (i.e., the fact `(served child)` is not in the state).
    3. Initialize the total heuristic cost to 0.
    4. For each unserved child:
        a. Determine the place where the child is waiting (`(waiting child place)`).
        b. Determine if the child is allergic to gluten (using static information).
        c. Find the "closest" available suitable sandwich for this child and add a corresponding cost to the total heuristic:
           - Cost 1: If there exists a suitable sandwich on *any* tray that is currently located *at the child's waiting place*. (Requires 1 `serve_sandwich` action).
           - Cost 2: Else, if there exists a suitable sandwich on *any* tray that is currently located *at a different place* than the child's waiting place. (Requires 1 `move_tray` + 1 `serve_sandwich` actions).
           - Cost 3: Else, if there exists a suitable sandwich currently *at the kitchen*. (Requires 1 `put_on_tray` + 1 `move_tray` + 1 `serve_sandwich` actions, assuming a tray is available at the kitchen).
           - Cost 4: Else, if a suitable sandwich *can be made* (i.e., a `notexist` sandwich object is available, and required components are at the kitchen). (Requires 1 `make_sandwich` + 1 `put_on_tray` + 1 `move_tray` + 1 `serve_sandwich` actions, assuming a tray is available at the kitchen).
           - Cost 1: If none of the above conditions are met (e.g., no `notexist` slot, or missing components if the simplified check fails). This child is unserved but blocked from the typical action sequence; we add a base cost of 1 to indicate it's still a goal to be achieved.
        d. A sandwich is "suitable" for an allergic child only if it is a no-gluten sandwich (`no_gluten_sandwich`). Any sandwich is suitable for a non-allergic child.
        e. A sandwich can be "made" if there is at least one `notexist` sandwich object available, and if the child is allergic, there is at least one no-gluten bread and one no-gluten content at the kitchen; otherwise (non-allergic child), there is at least one any bread and one any content at the kitchen.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.

        @param task: The planning task object.
        """
        self.goals = task.goals # Goals are needed to check if a state is a goal state (heuristic 0)
        static_facts = task.static

        # Extract all children and their allergy status from static facts
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.all_children = set()

        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.allergic_children.add(child)
                self.all_children.add(child)
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.not_allergic_children.add(child)
                self.all_children.add(child)

        # Extract no-gluten components from static facts
        self.ng_bread = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")}
        self.ng_content = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")}

    def is_suitable_sandwich(self, sandwich_name, child_name, state):
        """
        Checks if a given sandwich is suitable for a given child based on allergies.

        @param sandwich_name: The name of the sandwich object.
        @param child_name: The name of the child object.
        @param state: The current state (frozenset of facts).
        @return: True if the sandwich is suitable, False otherwise.
        """
        child_is_allergic = child_name in self.allergic_children

        if child_is_allergic:
            # Allergic children require a no-gluten sandwich
            return f"(no_gluten_sandwich {sandwich_name})" in state
        else:
            # Non-allergic children can have any sandwich
            return True

    def can_make_suitable_sandwich(self, child_name, state):
        """
        Checks if a suitable sandwich can potentially be made from available components
        and a notexist slot. Simplified check.

        @param child_name: The name of the child object.
        @param state: The current state (frozenset of facts).
        @return: True if a suitable sandwich can likely be made, False otherwise.
        """
        # Need a notexist slot
        has_notexist_slot = any(match(fact, "notexist", "*") for fact in state)
        if not has_notexist_slot:
            return False

        child_is_allergic = child_name in self.allergic_children

        if child_is_allergic:
            # Need NG bread and NG content at kitchen
            has_ng_bread_at_kitchen = any(match(fact, "at_kitchen_bread", b) for b in self.ng_bread if f"(at_kitchen_bread {b})" in state)
            has_ng_content_at_kitchen = any(match(fact, "at_kitchen_content", c) for c in self.ng_content if f"(at_kitchen_content {c})" in state)
            return has_ng_bread_at_kitchen and has_ng_content_at_kitchen
        else:
            # Need any bread and any content at kitchen
            has_any_bread_at_kitchen = any(match(fact, "at_kitchen_bread", "*") for fact in state)
            has_any_content_at_kitchen = any(match(fact, "at_kitchen_content", "*") for fact in state)
            return has_any_bread_at_kitchen and has_any_content_at_kitchen


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.

        @param node: The node containing the state.
        @return: The estimated cost to reach the goal state.
        """
        state = node.state

        # If the goal is reached, the heuristic is 0.
        if self.goals <= state:
            return 0

        total_heuristic_cost = 0

        # Find all unserved children and their waiting places
        unserved_children_info = {} # {child_name: waiting_place}
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        for child in self.all_children:
            if child not in served_children:
                # Find where the child is waiting
                waiting_place = None
                for fact in state:
                    if match(fact, "waiting", child, "*"):
                        waiting_place = get_parts(fact)[2]
                        break
                if waiting_place: # Child might not be waiting yet in some states, but goal is served
                     unserved_children_info[child] = waiting_place
                # If child is not served and not waiting, they still need to be served,
                # but the typical action sequence starts from waiting. We'll ignore
                # children not in a waiting state for this heuristic's calculation,
                # assuming the 'waiting' predicate is stable or only appears initially.
                # A more robust heuristic might count them with a base cost.
                # Based on the domain, children are always waiting until served.

        # Find available sandwiches and trays
        sandwiches_ontray = {} # {sandwich_name: tray_name}
        sandwiches_at_kitchen = set()
        trays_at_place = {} # {tray_name: place_name}
        notexist_sandwiches = set()
        no_gluten_sandwiches = set()

        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1:]
                sandwiches_ontray[s] = t
            elif match(fact, "at_kitchen_sandwich", "*"):
                s = get_parts(fact)[1]
                sandwiches_at_kitchen.add(s)
            elif match(fact, "at", "*", "*"):
                t, p = get_parts(fact)[1:]
                trays_at_place[t] = p
            elif match(fact, "notexist", "*"):
                s = get_parts(fact)[1]
                notexist_sandwiches.add(s)
            elif match(fact, "no_gluten_sandwich", "*"):
                 s = get_parts(fact)[1]
                 no_gluten_sandwiches.add(s)

        # Calculate cost for each unserved child
        for child, waiting_place in unserved_children_info.items():
            cost_for_child = 1 # Base cost: needs serving

            # Check if a suitable sandwich is already on a tray at the child's location
            found_suitable_at_location = False
            for s, t in sandwiches_ontray.items():
                if trays_at_place.get(t) == waiting_place:
                    if self.is_suitable_sandwich(s, child, state):
                        found_suitable_at_location = True
                        break

            if found_suitable_at_location:
                # Cost is 1 (serve)
                cost_for_child = 1
            else:
                # Needs delivery or creation
                cost_for_child += 1 # Cost for delivery/placement (move_tray)

                # Check if a suitable sandwich is on a tray elsewhere
                found_suitable_ontray_other = False
                for s, t in sandwiches_ontray.items():
                    if trays_at_place.get(t) is not None and trays_at_place[t] != waiting_place:
                         if self.is_suitable_sandwich(s, child, state):
                            found_suitable_ontray_other = True
                            break

                if found_suitable_ontray_other:
                    # Cost is 1 (serve) + 1 (move) = 2
                    cost_for_child = 2 # Override previous += 1
                else:
                    # Needs putting on tray or creation
                    cost_for_child += 1 # Cost for putting on tray

                    # Check if a suitable sandwich is at the kitchen
                    found_suitable_at_kitchen = False
                    for s in sandwiches_at_kitchen:
                        if self.is_suitable_sandwich(s, child, state):
                            found_suitable_at_kitchen = True
                            break

                    if found_suitable_at_kitchen:
                        # Cost is 1 (serve) + 1 (move) + 1 (put) = 3
                        cost_for_child = 3 # Override previous += 1
                    else:
                        # Needs making
                        cost_for_child += 1 # Cost for making

                        # Check if a suitable sandwich can be made
                        if self.can_make_suitable_sandwich(child, state):
                            # Cost is 1 (serve) + 1 (move) + 1 (put) + 1 (make) = 4
                            cost_for_child = 4 # Override previous += 1
                        else:
                            # Cannot make a suitable sandwich (e.g., no notexist slot, or missing components)
                            # This child is blocked from the standard sequence.
                            # Assign base cost 1 to indicate it's still an unserved goal.
                            cost_for_child = 1


            total_heuristic_cost += cost_for_child

        return total_heuristic_cost

