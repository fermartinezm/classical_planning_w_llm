from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and empty fact strings
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return [] # Return empty list for invalid format
    return fact[1:-1].split()

# Helper function to match PDDL facts (optional, but good practice)
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
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
    This heuristic estimates the number of actions needed to serve all passengers.
    It counts the number of 'board' actions, 'depart' actions, and estimates
    the minimum number of 'move' actions required for the lift to visit all
    necessary floors.

    # Assumptions
    - Floors are ordered linearly, defined by the `(above f_lower f_higher)` predicate.
    - Each 'board', 'depart', and 'move' action costs 1.
    - The lift can carry any number of passengers.
    - The heuristic does not consider the optimal sequence of stops, only the
      total vertical span of floors that need to be visited.

    # Heuristic Initialization
    - Parses `(above f_lower f_higher)` facts from `task.static` to build a
      mapping from floor names to their numerical rank (1 for the lowest floor, etc.).
    - Extracts the destination floor for each passenger mentioned in the goal
      from `task.static`.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check for Goal State: If the current state satisfies all goal conditions
       (all required passengers are served), the heuristic value is 0.

    2. Identify Unserved Passengers: Determine which passengers are not yet
       in the `(served p)` state.

    3. Count Board and Depart Actions:
       - For each unserved passenger `p` currently at their origin floor
         (`(origin p f_origin)` is true): This passenger needs one 'board'
         action and one 'depart' action. Increment the counts for both.
         Add their origin floor `f_origin` and destination floor `f_destin`
         to a set of floors the lift must visit.
       - For each unserved passenger `p` currently boarded (`(boarded p)`
         is true): This passenger needs one 'depart' action. Increment the
         'depart' count. Add their destination floor `f_destin` to the set
         of floors the lift must visit.

    4. Identify Relevant Floors for Movement: The set of floors the lift must
       visit includes all origin floors of waiting passengers and all destination
       floors of waiting or boarded passengers. Also include the lift's current
       floor in this set.

    5. Estimate Move Actions:
       - Map all relevant floor names to their numerical ranks using the
         `floor_to_rank` mapping created during initialization.
       - The estimated number of move actions is the difference between the
         maximum and minimum rank among all relevant floors. This represents
         the minimum vertical distance the lift must traverse to cover all
         necessary floors. If there are no relevant floors (which only happens
         in the goal state), the move cost is 0.

    6. Calculate Total Heuristic: The total heuristic value is the sum of the
       counted 'board' actions, the counted 'depart' actions, and the estimated
       'move' actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Facts that are not affected by actions.

        # Build floor order mapping: f_lower -> f_higher
        floor_above_map = {}
        all_floors = set()
        is_higher_in_above = set()  # Floors that appear as f_higher

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                # Assuming (above f_lower f_higher)
                f_lower, f_higher = parts[1], parts[2]
                floor_above_map[f_lower] = f_higher
                all_floors.add(f_lower)
                all_floors.add(f_higher)
                is_higher_in_above.add(f_higher)

        # Find the lowest floor (the one that is never f_higher)
        lowest_floor = None
        for floor in all_floors:
            if floor not in is_higher_in_above:
                lowest_floor = floor
                break

        # Build floor_to_rank mapping by traversing from the lowest floor
        self.floor_to_rank = {}
        if lowest_floor:
            current_floor = lowest_floor
            rank = 1
            # Traverse the chain f_min -> f_x -> ... -> f_max
            while current_floor in all_floors:
                 self.floor_to_rank[current_floor] = rank
                 if current_floor in floor_above_map:
                     current_floor = floor_above_map[current_floor]
                     rank += 1
                 else:
                     # Reached the highest floor in the chain
                     break
        # Note: If all_floors is empty or no clear lowest floor, floor_to_rank remains empty.
        # This should only happen for trivial or malformed problems.

        # Store goal locations for each passenger mentioned in the goal.
        self.passenger_destinations = {}
        passengers_in_goal = {get_parts(goal)[1] for goal in self.goals if get_parts(goal) and get_parts(goal)[0] == "served"}

        for passenger in passengers_in_goal:
             # Find the destination for this passenger in static facts
             for fact in static_facts:
                 fact_parts = get_parts(fact)
                 if fact_parts and fact_parts[0] == "destin" and fact_parts[1] == passenger:
                     self.passenger_destinations[passenger] = fact_parts[2]
                     break
             # If destination not found in static for a goal passenger, problem is ill-defined.
             # Assume it's always present.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Identify unserved passengers and their state (waiting or boarded)
        unserved_passengers = set()
        passenger_state = {} # {p: 'waiting' or 'boarded'}
        passenger_origin = {} # {p: f_origin}

        # Get lift's current floor
        lift_current_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at":
                lift_current_floor = parts[1]
                break

        # Collect info for unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact) and get_parts(fact)[0] == "served"}

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "origin":
                p, f_origin = parts[1], parts[2]
                # Only consider passengers that are part of the goal and not yet served
                if p in self.passenger_destinations and p not in served_passengers:
                    unserved_passengers.add(p)
                    passenger_state[p] = 'waiting'
                    passenger_origin[p] = f_origin
            elif parts and parts[0] == "boarded":
                p = parts[1]
                 # Only consider passengers that are part of the goal and not yet served
                if p in self.passenger_destinations and p not in served_passengers:
                    unserved_passengers.add(p)
                    passenger_state[p] = 'boarded'

        # Calculate board and depart actions needed
        board_actions_needed = 0
        depart_actions_needed = 0
        required_stop_floors = set()

        for p in unserved_passengers:
            destin_floor = self.passenger_destinations.get(p) # Get destination from pre-calculated map

            if passenger_state[p] == 'waiting':
                board_actions_needed += 1
                depart_actions_needed += 1
                origin_floor = passenger_origin[p]
                if origin_floor in self.floor_to_rank:
                    required_stop_floors.add(origin_floor)
                if destin_floor and destin_floor in self.floor_to_rank:
                    required_stop_floors.add(destin_floor)
            elif passenger_state[p] == 'boarded':
                depart_actions_needed += 1
                if destin_floor and destin_floor in self.floor_to_rank:
                    required_stop_floors.add(destin_floor)

        # Calculate estimated move actions
        estimated_move_actions = 0
        relevant_floor_ranks = set()

        if lift_current_floor and lift_current_floor in self.floor_to_rank:
             relevant_floor_ranks.add(self.floor_to_rank[lift_current_floor])

        for floor in required_stop_floors:
            # Floors were already checked against self.floor_to_rank when adding to required_stop_floors
            relevant_floor_ranks.add(self.floor_to_rank[floor])

        # If there are no relevant floors (only happens if all goal passengers are served,
        # which is handled by the initial goal check), move cost is 0.
        if relevant_floor_ranks:
            min_rank = min(relevant_floor_ranks)
            max_rank = max(relevant_floor_ranks)
            estimated_move_actions = max_rank - min_rank

        # Total heuristic is sum of action types
        total_cost = board_actions_needed + depart_actions_needed + estimated_move_actions

        return total_cost
