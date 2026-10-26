from fnmatch import fnmatch
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assuming Heuristic base class is available as heuristics.heuristic_base.Heuristic
# Replace the line below with the one above when integrating into the planner environment
class childsnackHeuristic:
# class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve all
    unserved children. It calculates the cost for each unserved child based on
    the current state of suitable sandwiches (gluten-free for allergic children,
    any for non-allergic) and their location (on tray at child's place, on tray
    elsewhere, in kitchen, or not yet made). The total heuristic is the sum
    of these individual child costs.

    # Assumptions
    - Bread and content portions are always available in the kitchen when needed
      to make a sandwich.
    - A 'notexist' sandwich object is always available when needed to make a
      sandwich.
    - A tray is always available in the kitchen when needed to put a sandwich on a tray.
    - Children remain waiting at their initial place until served.
    - The cost of moving a tray is 1 action, putting on a tray is 1 action,
      making a sandwich is 1 action, and serving is 1 action. These are unit costs.

    # Heuristic Initialization
    - Extracts the list of children that need to be served from the task goals.
    - Extracts the allergy status (allergic_gluten or not_allergic_gluten) for
      each child from the static facts.
    - Extracts the waiting place for each child from the initial state facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Identify all children that are in the goal state (i.e., need to be served).
    3. For each child that needs to be served:
        a. Check if the child is already served in the current state using the `(served ?c)` fact. If yes, the cost for this child is 0.
        b. If the child is not served, determine their waiting place and allergy status (retrieved during heuristic initialization).
        c. Determine the minimum estimated cost to serve this child based on the state of suitable sandwiches:
            - Start with a default cost of 4. This represents the maximum steps if a new sandwich is needed: make (1) + put on tray (1) + move tray (1) + serve (1).
            - Identify all sandwiches that currently exist (i.e., are not marked with the `notexist` predicate).
            - Filter the existing sandwiches to find those that are "suitable" for the current child based on their allergy status and the sandwich's gluten-free status (`no_gluten_sandwich`).
            - If no suitable sandwiches exist anywhere, the cost for this child remains 4.
            - If suitable sandwiches are found, check their locations to potentially reduce the cost:
                - If any suitable sandwich `S` is found on a tray `T` that is already at the child's waiting place `P` (i.e., `(ontray S T)` and `(at T P)` are in the state), the cost for this child is at least 1 (just the serve action). Update the child's cost to min(current_cost, 1).
                - Else, if any suitable sandwich `S` is found on a tray `T` that is *not* at the child's waiting place `P` (i.e., `(ontray S T)` is in the state, but `(at T P)` is not, and `(at T P_current)` for `P_current != P` is true), the cost is at least 2 (move tray + serve). Update the child's cost to min(current_cost, 2).
                - Else, if any suitable sandwich `S` is found in the kitchen (i.e., `(at_kitchen_sandwich S)` is in the state), the cost is at least 3 (put on tray + move tray + serve). Update the child's cost to min(current_cost, 3).
            - The child's cost is the minimum of the applicable cases (1, 2, 3, or the default 4).
        d. Add the calculated minimum cost for this child to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal children, allergy info, and waiting places."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Waiting facts are in initial state

        # Extract children that need to be served from goals
        self.goal_children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "served":
                self.goal_children.add(parts[1])

        # Extract allergy status for each child
        self.child_allergy = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "allergic_gluten":
                self.child_allergy[parts[1]] = True
            elif parts and parts[0] == "not_allergic_gluten":
                 self.child_allergy[parts[1]] = False

        # Extract waiting place for each child (from initial state as per examples)
        self.child_waiting_place = {}
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "waiting":
                 self.child_waiting_place[parts[1]] = parts[2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Build lookups from the current state for efficiency
        existing_sandwiches = set()
        sandwiches_in_kitchen = set()
        sandwiches_on_trays = {} # {sandwich: tray}
        trays_at_places = {} # {tray: place}
        gluten_free_sandwiches = set()
        served_children = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "notexist":
                # This sandwich does NOT exist yet, skip it from existing_sandwiches
                pass
            elif predicate == "at_kitchen_sandwich":
                s = parts[1]
                existing_sandwiches.add(s)
                sandwiches_in_kitchen.add(s)
            elif predicate == "ontray":
                s, t = parts[1], parts[2]
                existing_sandwiches.add(s)
                sandwiches_on_trays[s] = t
            elif predicate == "at": # Tray location
                 t, p = parts[1], parts[2]
                 trays_at_places[t] = p
            elif predicate == "no_gluten_sandwich":
                s = parts[1]
                gluten_free_sandwiches.add(s)
            elif predicate == "served":
                 c = parts[1]
                 served_children.add(c)


        # Calculate cost for each unserved child
        for child in self.goal_children:
            # Check if child is already served
            if child in served_children:
                continue # Child is served, cost is 0 for this child

            # Child is not served, calculate cost
            child_cost = 4 # Default: need to make sandwich, put, move, serve

            waiting_place = self.child_waiting_place.get(child)
            # Assuming valid problems have waiting facts for goal children in initial_state
            # If not found, child_cost remains 4, which is a safe upper bound.

            is_allergic = self.child_allergy.get(child)
            # Assuming valid problems have allergy facts for goal children in static_facts
            # If not found, child_cost remains 4, which is a safe upper bound.


            # Find suitable sandwiches among existing ones
            suitable_sandwiches = set()
            for s in existing_sandwiches:
                if is_allergic:
                    if s in gluten_free_sandwiches:
                        suitable_sandwiches.add(s)
                else:
                    suitable_sandwiches.add(s) # Any existing sandwich is suitable

            # If no suitable sandwich exists, cost remains 4.
            if not suitable_sandwiches:
                 total_cost += child_cost
                 continue # Move to next child

            # Check locations of suitable sandwiches to refine cost
            # Cost 1: Suitable sandwich S on tray T at waiting place P
            found_ready_to_serve = False
            for s in suitable_sandwiches:
                if s in sandwiches_on_trays:
                    t = sandwiches_on_trays[s]
                    if t in trays_at_places and trays_at_places[t] == waiting_place:
                        child_cost = min(child_cost, 1)
                        found_ready_to_serve = True
                        break # Found the best case for this child

            if found_ready_to_serve:
                 total_cost += child_cost
                 continue # Move to next child

            # Cost 2: Suitable sandwich S on tray T, but T is not at waiting place P
            found_on_tray_elsewhere = False
            for s in suitable_sandwiches:
                 if s in sandwiches_on_trays:
                      # Tray T exists, but we already know it's not at waiting_place
                      child_cost = min(child_cost, 2)
                      found_on_tray_elsewhere = True
                      break # Found the next best case

            if found_on_tray_elsewhere:
                 total_cost += child_cost
                 continue # Move to next child

            # Cost 3: Suitable sandwich S is in the kitchen
            found_in_kitchen = False
            for s in suitable_sandwiches:
                 if s in sandwiches_in_kitchen:
                      child_cost = min(child_cost, 3)
                      found_in_kitchen = True
                      break # Found the next best case

            if found_in_kitchen:
                 total_cost += child_cost
                 continue # Move to next child

            # If we reach here, it means suitable sandwiches exist, but they are
            # neither in the kitchen nor on any tray. This state shouldn't be
            # reachable through valid actions in this domain, as sandwiches
            # are always either at_kitchen_sandwich or ontray after being made.
            # The cost remains at its initial value (which was 4).
            total_cost += child_cost


        return total_cost
