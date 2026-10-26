from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential non-string input or malformed fact defensively
    if not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        # Assuming valid PDDL fact strings from the planner, this case might indicate an issue
        # but returning split parts of the inner string is a reasonable default behavior
        # for potentially malformed strings that still have content inside.
        # For strict PDDL facts, fact[1:-1] is safe.
        return fact[1:-1].split()
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It sums the minimum board/depart actions required for each unserved passenger
    and adds an estimate of the minimum lift movement needed to visit all relevant floors.

    # Assumptions
    - Actions (board, depart, up, down) have unit cost.
    - Floor names are structured such that their relative order can be determined
      from the `above` predicates, forming a single vertical sequence.
    - The heuristic assumes the lift can serve multiple passengers on a single trip,
      and estimates movement based on the range of floors requiring service.

    # Heuristic Initialization
    - Extracts destination floors for all passengers from static facts.
    - Builds a mapping from floor names to integer indices based on the `above` predicates
      to allow calculation of floor distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state (all passengers served). If yes, return 0.
    2. Identify the current floor of the lift.
    3. Categorize passengers into three groups based on the current state:
       - Waiting at origin floor (`origin` predicate).
       - Boarded in the lift (`boarded` predicate).
       - Already served (`served` predicate).
    4. Calculate the minimum number of board and depart actions required for all unserved passengers:
       - For each passenger not yet served: add 1 to the cost (for the final `depart` action).
       - For each passenger currently waiting at an origin floor: add an *additional* 1 to the cost (for the initial `board` action).
    5. Identify the set of floors that require a visit by the lift:
       - The origin floor for every waiting passenger.
       - The destination floor for every waiting passenger (needed after boarding).
       - The destination floor for every boarded passenger.
    6. If there are no floors requiring a visit (meaning no unserved passengers), return the action count calculated in step 4 (which will be 0).
    7. Convert the required floor names to their integer indices and find the minimum (`min_req_floor_int`) and maximum (`max_req_floor_int`) indices among them.
    8. Estimate the minimum movement cost to visit all required floors. This is estimated as the minimum moves to reach either the lowest or highest required floor from the current lift floor, plus the total vertical distance between the lowest and highest required floors (`max_req_floor_int - min_req_floor_int`). This approximates the cost of one sweep covering the necessary range.
    9. The total heuristic value is the sum of the minimum board/depart actions (step 4) and the estimated movement cost (step 8).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and floor ordering.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract passenger destinations and collect all passenger names
        self.destin_map = {}
        self.all_passengers = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'destin':
                # parts[1] is passenger, parts[2] is destination_floor
                self.destin_map[parts[1]] = parts[2]
                self.all_passengers.add(parts[1])

        # Build floor mapping (name to integer index)
        above_relations = [] # Stores (floor_above, floor_below) tuples
        all_floors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'above':
                above_relations.append((parts[1], parts[2]))
                all_floors.add(parts[1])
                all_floors.add(parts[2])

        # Map floor_below -> floor_above
        floor_above_map = {below: above for above, below in above_relations}

        # Find the lowest floor (a floor that is never the first argument of 'above')
        floors_that_are_above_others = set(above for above, below in above_relations)
        lowest_floor = None
        # Iterate through all floors to find the one not appearing as the 'above' floor
        for f in all_floors:
            if f not in floors_that_are_above_others:
                lowest_floor = f
                break

        # Build the ordered floor list and mapping
        self.floor_to_int = {}
        self.int_to_floor = {}
        current_floor = lowest_floor
        floor_index = 1
        while current_floor:
            self.floor_to_int[current_floor] = floor_index
            self.int_to_floor[floor_index] = current_floor
            floor_index += 1
            # Move to the floor directly above the current one
            current_floor = floor_above_map.get(current_floor)

        self.num_floors = len(self.floor_to_int)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state

        # Check if goal is reached
        # The goal is defined by task.goals, which should contain (served ?p) for all passengers.
        # We can check this directly.
        if self.goals <= state:
            return 0

        # Get current lift location
        lift_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'lift-at':
                lift_location = parts[1]
                break
        # Assuming lift_at fact is always present in a valid state
        lift_floor_int = self.floor_to_int[lift_location]

        # Identify passengers and their states
        passengers_waiting = {} # passenger -> origin_floor
        passengers_boarded = set() # passenger
        passengers_served = set() # passenger

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'origin':
                passengers_waiting[parts[1]] = parts[2]
            elif parts and parts[0] == 'boarded':
                passengers_boarded.add(parts[1])
            elif parts and parts[0] == 'served':
                passengers_served.add(parts[1])

        # Passengers not yet served are those in self.all_passengers minus those served
        passengers_not_served = self.all_passengers - passengers_served

        # Heuristic calculation
        total_cost = 0 # This will accumulate action costs

        # Cost for board/depart actions
        # Each passenger not served needs a final 'depart' action.
        # Each passenger waiting needs an additional 'board' action before departing.
        for p in passengers_not_served:
            total_cost += 1 # Cost for the final 'depart' action
            if p in passengers_waiting:
                total_cost += 1 # Additional cost for the 'board' action

        # Identify floors that need to be visited
        required_floors = set()
        for p in passengers_not_served:
            # Need to visit destination floor for all unserved passengers
            # Ensure the passenger's destination is known (should be from static facts)
            if p in self.destin_map:
                 required_floors.add(self.destin_map[p])
            # Need to visit origin floor for waiting passengers
            if p in passengers_waiting:
                required_floors.add(passengers_waiting[p])

        # If no floors need visiting (i.e., all passengers are served),
        # the action cost is 0 and movement cost is 0.
        if not required_floors:
             return total_cost # This will be 0 if passengers_not_served is empty

        # Estimate movement cost
        required_floor_ints = sorted([self.floor_to_int[f] for f in required_floors])
        min_req_floor_int = required_floor_ints[0]
        max_req_floor_int = required_floor_ints[-1]

        # Minimum moves to reach the range [min_req, max_req] from current lift location
        # plus the moves to sweep through the range.
        # Option 1: Go to min_req, then sweep to max_req
        moves_option1 = abs(lift_floor_int - min_req_floor_int) + (max_req_floor_int - min_req_floor_int)
        # Option 2: Go to max_req, then sweep to min_req
        moves_option2 = abs(lift_floor_int - max_req_floor_int) + (max_req_floor_int - min_req_floor_int)

        movement_cost_estimate = min(moves_option1, moves_option2)

        # Total heuristic is sum of action costs and movement cost
        total_cost += movement_cost_estimate

        return total_cost
