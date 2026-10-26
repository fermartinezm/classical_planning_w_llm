from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing
class Heuristic:
    def __init__(self, task):
        self.task = task
        pass
    def __call__(self, node):
        pass

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or invalid format gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers. It sums the minimum actions (board/depart) for each unserved
    passenger and adds an estimate of the movement cost required for each
    passenger's individual journey, starting from the lift's current position.
    It assumes passengers are transported one by one, which might overestimate
    movement but provides a potentially informative guide for greedy search.

    # Assumptions
    - Floors are typically totally ordered by the 'above' predicate, forming a linear chain. A fallback to alphabetical sorting is used if the chain is not clear.
    - Lift capacity is sufficient (ignored by this heuristic).
    - Actions (board, depart, up, down) have a cost of 1.
    - All unserved passengers are either waiting at their origin or are boarded.
    - All passengers mentioned in the goal have a 'destin' fact in the static information.
    - All floors mentioned in relevant static facts ('above', 'destin') or initial state ('lift-at', 'origin') can be ordered.

    # Heuristic Initialization
    - Extract the destination floor for each passenger from the static facts, based on the passengers mentioned in the goal.
    - Determine the total order of floors from 'above' static facts. If a clear linear chain is found, create mappings between floor names and numerical indices based on this chain. Otherwise, fall back to alphabetical sorting of all relevant floors found in static facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Check if the goal is already reached. If yes, the heuristic is 0.
    2.  Identify the lift's current floor by finding the fact `(lift-at ?f)` in the state. If not found, return infinity.
    3.  Identify the set of passengers that still need to be served by checking which `(served ?p)` goal facts are *not* present in the current state.
    4.  For efficiency, pre-process the state to quickly find the origin floor for waiting passengers `(origin ?p ?f)` and the set of boarded passengers `(boarded ?p)`.
    5.  Initialize the total heuristic cost to 0.
    6.  Iterate through the set of passengers that need to be served (identified in step 3).
    7.  For each unserved passenger `p`:
        a.  Retrieve their destination floor `f_destin` (pre-calculated in `__init__`). If destination is unknown (not in static facts), return infinity.
        b.  If the passenger `p` is found in the set of waiting passengers (from step 4):
            -   Retrieve their origin floor `f_origin`. If origin is unknown (not in state), this case shouldn't happen for unserved passengers in valid states.
            -   Add 2 to the total cost (representing the `board` and `depart` actions needed for this passenger).
            -   Calculate the estimated movement cost for this passenger's journey: `abs(index(current_lift_floor) - index(f_origin)) + abs(index(f_origin) - index(f_destin))`. Add this movement cost to the total cost. If origin or destination index is unknown, return infinity.
        c.  If the passenger `p` is found in the set of boarded passengers (from step 4):
            -   Add 1 to the total cost (representing the `depart` action needed for this passenger).
            -   Calculate the estimated movement cost for this passenger's journey: `abs(index(current_lift_floor) - index(f_destin))`. Add this movement cost to the total cost. If destination index is unknown, return infinity.
        d.  (Passengers who are unserved but neither waiting nor boarded should not occur in valid states; this case is implicitly handled by the checks above).
    8.  Return the total calculated cost.

    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal destinations for passengers mentioned in the goals.
        self.passenger_destins = {}
        all_goal_passengers = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "served" and len(args) == 1:
                all_goal_passengers.add(args[0])

        # Find destinations for all passengers from static facts
        # (Assuming all passengers in goals have a destin fact in static)
        all_relevant_floors = set()
        for fact in static_facts:
             predicate, *args = get_parts(fact)
             if predicate == "destin" and len(args) == 2 and args[0] in all_goal_passengers:
                 self.passenger_destins[args[0]] = args[1]
                 all_relevant_floors.add(args[1]) # Add destination floor

        # Determine floor order and create index mappings
        above_map = {} # maps floor_below -> floor_above
        floors_that_are_above_something = set() # Floors that appear as the first arg in (above f1 f2)

        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "above" and len(args) == 2:
                f_above, f_below = args
                above_map[f_below] = f_above
                all_relevant_floors.add(f_above)
                all_relevant_floors.add(f_below)
                floors_that_are_above_something.add(f_above)

        # Find the lowest floor: a floor in all_relevant_floors that is not the VALUE in any above_map entry
        lowest_floor = None
        potential_lowest_floors = all_relevant_floors - floors_that_are_above_something

        ordered_floors = []
        if len(potential_lowest_floors) == 1:
             lowest_floor = list(potential_lowest_floors)[0]
             # Build the ordered list of floors starting from the lowest
             current_floor = lowest_floor
             # Follow the chain upwards
             while current_floor is not None and current_floor in all_relevant_floors: # Add check for safety
                 ordered_floors.append(current_floor)
                 current_floor = above_map.get(current_floor) # Get the floor immediately above

        # If the chain didn't include all relevant floors, or no unique lowest floor was found,
        # fall back to alphabetical sort of all relevant floors.
        if len(ordered_floors) != len(all_relevant_floors):
             sorted_floors = sorted(list(all_relevant_floors))
             self.floor_to_index = {f: i for i, f in enumerate(sorted_floors)}
             self.index_to_floor = {i: f for i, f in enumerate(sorted_floors)}
             # print(f"Warning: Floor ordering from 'above' facts incomplete or failed. Assuming alphabetical order for {sorted_floors}")
        else:
             self.floor_to_index = {f: i for i, f in enumerate(ordered_floors)}
             self.index_to_floor = {i: f for i, f in enumerate(ordered_floors)}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached first for efficiency and edge cases
        all_served = True
        for passenger in self.passenger_destins.keys():
            if f"(served {passenger})" not in state:
                all_served = False
                break
        if all_served:
            return 0

        # If floor indexing failed completely and goal is not met, return infinity
        if not self.floor_to_index:
             return float('inf')


        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*") and len(get_parts(fact)) == 2:
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown and goal is not met, return infinity
        if current_lift_floor is None:
             return float('inf')

        current_lift_index = self.floor_to_index.get(current_lift_floor)

        # If current lift floor is not in our floor index map (shouldn't happen in valid problem)
        if current_lift_index is None:
             # This indicates a floor in the state wasn't in static facts used for indexing.
             # Return infinity if not goal (already checked).
             return float('inf')


        # Pre-process state to find passenger statuses
        waiting_passengers = {} # p -> origin_floor
        boarded_passengers = set()
        # served_passengers = set() # Not needed if we check against goal facts not in state

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts
            predicate = parts[0]
            if predicate == "origin" and len(parts) == 3:
                p, f = parts[1:]
                waiting_passengers[p] = f
            elif predicate == "boarded" and len(parts) == 2:
                p = parts[1]
                boarded_passengers.add(p)
            # elif predicate == "served" and len(parts) == 2:
            #      p = parts[1]
            #      served_passengers.add(p) # Not needed

        total_cost = 0

        # Iterate through all passengers that need to be served eventually (those in goals)
        for passenger in self.passenger_destins.keys():
            # Check if the passenger is already served in the current state
            # This check is done at the beginning for efficiency, but double-check here
            if f"(served {passenger})" in state:
                 continue

            # Passenger is unserved. Determine their current status.
            destin_floor = self.passenger_destins[passenger]
            destin_index = self.floor_to_index.get(destin_floor)

            # If destination floor is not in our map (shouldn't happen in valid problem)
            if destin_index is None:
                 # Cannot compute cost for this passenger, return infinity
                 return float('inf')


            if passenger in waiting_passengers:
                # Passenger is waiting at origin
                origin_floor = waiting_passengers[passenger]
                origin_index = self.floor_to_index.get(origin_floor)

                # If origin floor is not in our map (shouldn't happen in valid problem)
                if origin_index is None:
                     return float('inf')

                # Cost: board (1) + depart (1) + move(current -> origin) + move(origin -> destin)
                cost_for_p = 2
                cost_for_p += abs(current_lift_index - origin_index)
                cost_for_p += abs(origin_index - destin_index)
                total_cost += cost_for_p

            elif passenger in boarded_passengers:
                # Passenger is boarded
                # Cost: depart (1) + move(current -> destin)
                cost_for_p = 1
                cost_for_p += abs(current_lift_index - destin_index)
                total_cost += cost_for_p

            # Else: Passenger is unserved but neither waiting nor boarded.
            # This implies an invalid state according to the domain structure.
            # We assume valid states derived from the initial state and actions.
            # If this case is reached, the state is likely malformed.
            # For robustness, we could return infinity or raise an error.
            # Assuming valid states, this 'else' block is not needed.


        return total_cost
