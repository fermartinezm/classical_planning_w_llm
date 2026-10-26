from heuristics.heuristic_base import Heuristic
# No need for fnmatch as we are parsing facts manually

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the number of 'board' actions needed for waiting passengers, the number
        of 'depart' actions needed for boarded passengers, and an estimate of
        the movement cost required to visit all necessary floors. Necessary
        floors include those where unserved passengers are waiting for pickup
        and those where boarded unserved passengers need to be dropped off.
        It is designed for greedy best-first search and is not admissible.

    Assumptions:
        - Floor names follow a consistent naming convention (e.g., f1, f2, ...).
        - The 'above' predicates define a strict linear ordering of floors,
          forming a single chain.
        - Passenger destinations are static and provided in the initial state/static facts.
        - The goal is always a conjunction of '(served ?p)' facts for some passengers.

    Heuristic Initialization:
        In the constructor, the heuristic processes the static facts to:
        1. Map each passenger to their destination floor using the 'destin' predicates.
           This information is stored in `self.destinations`.
        2. Determine the linear ordering of floors based on the 'above' predicates.
           It finds the lowest floor (a floor that is the second argument of an
           'above' predicate but never the first argument) and then follows the
           chain of 'above' relations to build an ordered list of floors.
           Mappings between floor names (e.g., 'f1') and their corresponding
           integer levels (e.g., 1) are created and stored in `self.floor_to_level`
           and `self.level_to_floor`. The total number of floors is stored in
           `self.num_floors`.
        3. Identify the set of passengers that need to be served according to the
           goal state. This set is stored in `self.goal_passengers`.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state (represented as a frozenset of fact strings):
        1. Check if the current state is the goal state by verifying if all facts
           in `self.goals` are present in the state. If it is the goal state,
           the heuristic value is 0.
        2. Identify the set of unserved passengers that are relevant to the goal.
           These are the passengers in `self.goal_passengers` for whom the fact
           '(served ?p)' is not present in the current state.
        3. If there are no unserved goal passengers, but the state was not
           identified as a goal state in step 1, this indicates an inconsistency
           (which shouldn't occur in valid miconic problems where goals are only
           '(served p)'). However, the check in step 1 is sufficient. If we reach
           this point, there is at least one unserved goal passenger.
        4. Iterate through the facts in the current state to extract necessary information:
           a. Find the current floor of the lift from the '(lift-at ?f)' fact.
           b. Identify unserved passengers who are currently waiting at their
              origin floor (indicated by an '(origin ?p ?f)' fact for an unserved ?p).
              Count these passengers (`passengers_waiting_count`) and collect
              the levels of their origin floors (`required_pickup_levels`).
           c. Identify unserved passengers who are currently boarded (indicated
              by a '(boarded ?p)' fact for an unserved ?p). Count these passengers
              (`passengers_boarded_count`) and collect the levels of their
              destination floors (using `self.destinations`) into
              `required_dropoff_levels`.
        5. The initial heuristic value `h` is set to the sum of
           `passengers_waiting_count` and `passengers_boarded_count`. This
           accounts for the minimum number of 'board' and 'depart' actions needed.
        6. Combine the sets of required pickup and dropoff floor levels into a
           single set `required_levels`.
        7. If `required_levels` is not empty:
           a. Find the minimum (`min_level`) and maximum (`max_level`) floor
              levels among the `required_levels`.
           b. Get the level of the current lift floor (`current_level`).
           c. Calculate an estimated movement cost (`movement_h`). This estimate
              is the distance needed to traverse the full span of required levels
              (`max_level - min_level`) plus the distance from the current lift
              level to the closest edge of this span. The formula used is
              `(max_level - min_level) + max(0, min_level - current_level) + max(0, current_level - max_level)`.
           d. Add `movement_h` to the heuristic value `h`.
        8. Return the final heuristic value `h`.
    """
    def __init__(self, task):
        super().__init__(task) # Call parent constructor if needed, though Heuristic base might not have one
        self.goals = task.goals
        static_facts = task.static

        self.destinations = {}
        above_pairs = [] # List of (higher_floor, lower_floor)
        all_floors = set()
        higher_floors_set = set()
        lower_floors_set = set()

        # Parse static facts to get destinations and floor ordering
        for fact in static_facts:
            parts = self.get_parts(fact)
            if parts[0] == 'destin':
                # Store destination mapping: passenger -> floor
                self.destinations[parts[1]] = parts[2]
            elif parts[0] == 'above':
                # Store above relations: (higher_floor, lower_floor)
                higher, lower = parts[1], parts[2]
                above_pairs.append((higher, lower))
                all_floors.add(higher)
                all_floors.add(lower)
                higher_floors_set.add(higher)
                lower_floors_set.add(lower)

        # Build floor levels based on 'above' relations
        self.floor_to_level = {}
        self.level_to_floor = {}

        if not all_floors:
            # Handle case with no floors (should not happen in valid problems)
            self.num_floors = 0
        else:
            # Find the lowest floor: it's in lower_floors_set but not in higher_floors_set
            # Assumes a single linear chain of floors. If only one floor, it's the lowest.
            potential_lowest = list(lower_floors_set - higher_floors_set)
            if not potential_lowest:
                 # This case handles a single floor or potentially malformed above facts
                 if len(all_floors) == 1:
                     lowest_floor = list(all_floors)[0]
                 else:
                     # This indicates an issue with above facts not defining a clear chain
                     # For robustness, could pick an arbitrary floor or raise error
                     # Assuming valid miconic domains define a chain, this shouldn't be reached
                     print("Warning: Could not determine unique lowest floor from 'above' facts. Using arbitrary floor.")
                     lowest_floor = list(all_floors)[0] # Fallback

            else:
                lowest_floor = potential_lowest[0] # Assumes exactly one lowest

            # Build the chain: map lower floor to higher floor
            floor_above_map = {lower: higher for higher, lower in above_pairs}

            current_floor = lowest_floor
            level = 1
            while current_floor is not None:
                self.floor_to_level[current_floor] = level
                self.level_to_floor[level] = current_floor
                level += 1
                # Find the floor directly above the current one
                current_floor = floor_above_map.get(current_floor)

            self.num_floors = level - 1 # Total number of floors

        # Identify passengers that need to be served based on goals
        # Assuming goals are always of the form (served ?p)
        self.goal_passengers = {self.get_parts(g)[1] for g in self.goals if self.get_parts(g)[0] == 'served'}


    def get_parts(self, fact: str) -> list[str]:
        """Helper to split a PDDL fact string into predicate and arguments."""
        # Assumes fact is like '(predicate arg1 arg2)'
        # Removes leading '(' and trailing ')' and splits by space
        return fact[1:-1].split()

    def __call__(self, node) -> int:
        """
        Computes the domain-dependent heuristic value for a given state.

        Args:
            node: The search node containing the state.

        Returns:
            The estimated number of actions to reach the goal state.
        """
        state = node.state

        # 1. Check for goal state
        if self.goals <= state:
             return 0

        # 2. Identify unserved passengers relevant to the goal
        unserved_passengers_set = {p for p in self.goal_passengers if '(served ' + p + ')' not in state}

        # If no unserved goal passengers, but not goal state? This shouldn't happen
        # in a valid miconic problem definition where goals are only (served p).
        # The check `if self.goals <= state` handles the true goal state.
        # If we reach here, unserved_passengers_set must be non-empty.

        # 3. Count unserved waiting and boarded passengers and find required floors
        passengers_waiting_count = 0
        passengers_boarded_count = 0
        required_pickup_levels = set()
        required_dropoff_levels = set()
        current_lift_floor = None

        for fact in state:
            parts = self.get_parts(fact)
            if parts[0] == 'lift-at':
                current_lift_floor = parts[1]
            elif parts[0] == 'origin':
                p, f = parts[1], parts[2]
                # Check if this passenger is one of the unserved goal passengers
                if p in unserved_passengers_set:
                    passengers_waiting_count += 1
                    # Ensure floor exists in our mapping (should always be true for valid problems)
                    if f in self.floor_to_level:
                        required_pickup_levels.add(self.floor_to_level[f])
            elif parts[0] == 'boarded':
                p = parts[1]
                # Check if this passenger is one of the unserved goal passengers
                if p in unserved_passengers_set:
                    passengers_boarded_count += 1
                    # Get destination from pre-calculated static info
                    dest_floor = self.destinations.get(p)
                    # Ensure destination exists and is in floor mapping
                    if dest_floor and dest_floor in self.floor_to_level:
                         required_dropoff_levels.add(self.floor_to_level[dest_floor])
                    # else: Should not happen for goal passengers in valid problem

        # 4. Heuristic starts with board/depart action counts
        # Each waiting passenger needs a 'board' action.
        # Each boarded passenger needs a 'depart' action.
        h = passengers_waiting_count + passengers_boarded_count

        # 5. Identify all required floor levels (union of pickup and dropoff levels)
        required_levels = required_pickup_levels | required_dropoff_levels

        # 6. Calculate and add movement cost
        if required_levels:
            min_level = min(required_levels)
            max_level = max(required_levels)

            # Ensure current_lift_floor was found and is in mapping
            if current_lift_floor and current_lift_floor in self.floor_to_level:
                current_level = self.floor_to_level[current_lift_floor]

                # Movement estimate: distance to reach the span + distance to traverse the span
                # Distance to reach the span [min_level, max_level] from current_level
                dist_to_span = 0
                if current_level < min_level:
                    dist_to_span = min_level - current_level
                elif current_level > max_level:
                    dist_to_span = current_level - max_level

                # Distance to traverse the span [min_level, max_level]
                span_dist = max_level - min_level

                # Total estimated movement cost
                movement_h = dist_to_span + span_dist
                h += movement_h
            # else: Should not happen in a valid state representation where lift-at is present

        # 7. Return total heuristic value
        return h
