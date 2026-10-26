from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and remove surrounding parentheses
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Handle cases that might not be standard facts if necessary, or raise an error
    # For this domain, facts are expected to be strings like '(predicate arg1 arg2)'
    return [] # Or raise an error

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the remaining effort by summing up the minimum
    estimated steps required to serve each unserved child. The estimate for
    each child is based on the most "ready" suitable sandwich available
    in the current state.

    # Assumptions
    - Each child requires exactly one sandwich to be served.
    - All necessary ingredients (bread, content) are assumed to be available
      in the kitchen if a sandwich needs to be made.
    - Trays are available and can be moved freely between places.
    - A non-allergic child can be served any type of sandwich (gluten or no-gluten).
    - An allergic child can only be served a no-gluten sandwich.
    - The cost of each individual action (make, put, move, serve) is 1.
    - The heuristic calculation for each child is independent, ignoring
      potential resource conflicts (like multiple children needing the same
      tray or the same ingredients simultaneously).

    # Heuristic Initialization
    - Extracts static information from the task:
        - Which children are allergic and which are not.
        - The waiting place for each child.
        - The set of all child objects and sandwich objects defined in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of the estimated costs for each child
    who is not yet served in the current state.

    For each unserved child `C`:
    1.  Determine the child's waiting place `P` and allergy status.
    2.  Identify the type of sandwich required: no-gluten if allergic, any
        existing sandwich if not allergic.
    3.  Find the "most ready" suitable sandwich available in the current state.
        A sandwich `S` is "available" if `(notexist S)` is *not* in the state.
        A sandwich `S` is "suitable" if it matches the child's allergy needs.
    4.  Estimate the minimum remaining steps for child `C` based on the state
        of the most ready suitable sandwich found:
        -   If a suitable sandwich `S` exists on a tray `T` that is already
            at the child's place `P` (`(ontray S T)` and `(at T P)` are true):
            Estimated cost for this child = 1 (serve).
        -   Else, if a suitable sandwich `S` exists on *any* tray `T`
            (`(ontray S T)` is true for some `T`):
            Estimated cost for this child = 2 (move tray to P, serve).
        -   Else, if a suitable sandwich `S` exists in the kitchen
            (`(at_kitchen_sandwich S)` is true):
            Estimated cost for this child = 3 (put on tray, move tray to P, serve).
        -   Else (no suitable sandwich exists anywhere):
            Estimated cost for this child = 4 (make sandwich, put on tray, move tray to P, serve).
    5.  Sum the estimated costs for all unserved children to get the total
        heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.

        @param task: The planning task object containing initial state, goals, and static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Extract all objects by parsing initial and static facts
        all_objects = set()
        for fact in initial_state | static_facts:
             parts = get_parts(fact)
             if parts: # Ensure parts is not empty
                 # All parts after the predicate are objects/parameters
                 all_objects.update(parts[1:])

        # Categorize objects based on naming convention (common in benchmarks)
        self.children = {obj for obj in all_objects if obj.startswith('child')}
        self.sandwiches = {obj for obj in all_objects if obj.startswith('sandw')}
        self.trays = {obj for obj in all_objects if obj.startswith('tray')}
        self.places = {obj for obj in all_objects if obj.startswith('table') or obj == 'kitchen'}


        # Store child allergy status and waiting places from static facts
        self.child_allergy = {} # child -> True if allergic, False otherwise
        self.child_place = {}   # child -> waiting_place

        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.child_allergy[child] = True
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.child_allergy[child] = False
            elif match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:]
                self.child_place[child] = place

        # Ensure all children found have allergy status and place (should be true based on domain/instance structure)
        # For robustness, one might add checks or default values here.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state (all children served).

        @param node: The current state node in the search tree.
        @return: The estimated heuristic cost.
        """
        state = node.state  # Current world state as a frozenset of fact strings.

        total_cost = 0  # Initialize the total heuristic cost.

        # Identify children who are not yet served
        unserved_children = {
            child for child in self.children
            if '(served ' + child + ')' not in state
        }

        # Find all sandwiches that currently exist (i.e., are not marked as notexist)
        existing_sandwiches = {
            sandw for sandw in self.sandwiches
            if '(notexist ' + sandw + ')' not in state
        }

        # Find which existing sandwiches are no-gluten
        no_gluten_sandwiches_exist = {
            sandw for sandw in existing_sandwiches
            if '(no_gluten_sandwich ' + sandw + ')' in state
        }

        # Find locations of existing sandwiches and trays
        sandwich_location = {} # sandw -> 'kitchen' or tray_name
        tray_location = {}     # tray -> place_name

        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                sandw = get_parts(fact)[1]
                if sandw in existing_sandwiches: # Only track existing ones
                    sandwich_location[sandw] = 'kitchen'
            elif match(fact, "ontray", "*", "*"):
                sandw, tray = get_parts(fact)[1:]
                if sandw in existing_sandwiches: # Only track existing ones
                     sandwich_location[sandw] = tray
            elif match(fact, "at", "*", "*"):
                 obj, place = get_parts(fact)[1:]
                 if obj in self.trays: # Only track trays
                     tray_location[obj] = place

        # Calculate cost for each unserved child
        for child in unserved_children:
            child_place = self.child_place.get(child)
            is_allergic = self.child_allergy.get(child, False) # Default to False if status missing

            # Find suitable existing sandwiches for this child
            if is_allergic:
                suitable_sandwiches = no_gluten_sandwiches_exist
            else:
                # Non-allergic children can eat any existing sandwich
                suitable_sandwiches = existing_sandwiches

            min_child_cost = 4 # Default worst case: make, put, move, serve

            if not suitable_sandwiches:
                 # No suitable sandwich exists at all
                 min_child_cost = 4
            else:
                 # Check the state of the most ready suitable sandwich
                 found_ready_at_place = False
                 found_on_any_tray = False
                 found_in_kitchen = False

                 for sandw in suitable_sandwiches:
                     current_loc = sandwich_location.get(sandw)

                     if current_loc and current_loc != 'kitchen': # It's on a tray
                         tray = current_loc
                         if tray_location.get(tray) == child_place:
                             found_ready_at_place = True
                             break # Found the best case for this child
                         else:
                             found_on_any_tray = True
                     elif current_loc == 'kitchen':
                         found_in_kitchen = True

                 if found_ready_at_place:
                     min_child_cost = 1 # Serve
                 elif found_on_any_tray:
                     min_child_cost = 2 # Move tray, serve
                 elif found_in_kitchen:
                     min_child_cost = 3 # Put on tray, move tray, serve
                 else:
                     # Suitable sandwiches exist, but they are not in kitchen or on any tray?
                     # This case shouldn't typically happen in a valid state if the above logic is correct.
                     # It might imply a sandwich was served or is in an unexpected state.
                     # Given the action model, sandwiches are either notexist, at_kitchen_sandwich, or ontray.
                     # If a suitable sandwich exists but isn't in one of these states, it was likely served.
                     # However, the child is unserved, so this sandwich must be for someone else or an error state.
                     # We default to the worst case if we can't find a suitable sandwich in a known location.
                     min_child_cost = 4 # Should ideally not be reached if suitable_sandwiches is not empty

            total_cost += min_child_cost

        return total_cost

