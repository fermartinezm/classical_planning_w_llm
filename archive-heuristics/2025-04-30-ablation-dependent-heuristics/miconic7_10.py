class miconicHeuristic:
    """
    A domain-dependent heuristic for the Miconic domain.
    """

    def __init__(self, task):
        """
        Initializes the miconic heuristic.

        Heuristic Initialization:
        Processes the static information from the task to build data structures
        needed for heuristic computation. This includes:
        1. Mapping passenger names to their destination floor names.
        2. Determining the linear order of floors and creating mappings
           between floor names and integer indices.
        3. Identifying the set of all passengers in the problem.
        """
        self.passenger_destinations = {}
        self.floor_above_map = {}
        self.floor_below_map = {}
        all_floor_names = set()

        # Parse static facts
        for fact_str in task.static:
            # Fact format: '(predicate arg1 arg2 ...)'
            # Extract predicate and arguments by splitting the string inside the brackets
            parts = fact_str[1:-1].split(' ')
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'destin':
                p, f = args
                self.passenger_destinations[p] = f
            elif predicate == 'above':
                f1, f2 = args # f1 is above f2
                self.floor_above_map[f2] = f1
                self.floor_below_map[f1] = f2
                all_floor_names.add(f1)
                all_floor_names.add(f2)

        # Build floor order and index maps
        self.floor_to_index = {}
        self.index_to_floor = {}
        if all_floor_names:
            # Find the lowest floor: a floor f such that no other floor is below f.
            # This means f is never the second argument of 'above', i.e., never a value in floor_below_map.
            lowest_floor = None
            # Get the set of all floors that are below some other floor
            floors_that_are_below_another = set(self.floor_below_map.values())
            # The lowest floor is one that is in all_floor_names but not in floors_that_are_below_another
            for f in all_floor_names:
                if f not in floors_that_are_below_another:
                    lowest_floor = f
                    break # Assuming a unique lowest floor in a connected structure

            if lowest_floor:
                ordered_floors = []
                curr = lowest_floor
                # Traverse upwards using the floor_above_map
                while curr is not None:
                    ordered_floors.append(curr)
                    curr = self.floor_above_map.get(curr) # Get floor above curr

                # Create index mappings
                self.floor_to_index = {f: i for i, f in enumerate(ordered_floors)}
                self.index_to_floor = {i: f for i, f in enumerate(ordered_floors)}
            # else: Could not find a unique lowest floor? Problematic instance.
        # else: No floors found? Problematic instance.

        # Identify all passengers mentioned in the problem (from destinations)
        self.all_passengers = set(self.passenger_destinations.keys())


    def __call__(self, state):
        """
        Computes the domain-dependent heuristic value for the given state.

        Summary:
        This heuristic estimates the remaining cost to reach the goal state
        by summing the number of discrete actions (board/depart) needed for
        unserved passengers and the estimated lift movement cost to visit
        all necessary floors (origins of waiting passengers, destinations
        of boarded passengers).

        Assumptions:
        - The PDDL state representation follows the expected string format
          '(predicate arg1 arg2 ...)'.
        - The static facts define a linear, connected sequence of floors
          using the 'above' predicate.
        - Passengers are either waiting at their origin, boarded, or served.
          An unserved passenger is either waiting or boarded.

        Step-By-Step Thinking for Computing Heuristic:
        1. Parse the current state to identify:
           - The current floor of the lift.
           - Passengers waiting at their origin floors.
           - Passengers currently boarded in the lift.
           - Passengers already served.
        2. Determine the set of unserved passengers by subtracting served
           passengers from the total set of passengers in the problem.
        3. Calculate the number of discrete actions needed:
           - For each unserved passenger:
             - If the passenger is waiting at their origin, they need a 'board' action (add 1 to action count).
             - If the passenger is currently boarded, they need a 'depart' action (add 1 to action count).
        4. Identify the set of 'required floors' that the lift must visit:
           - For each unserved passenger waiting at their origin floor, add their origin floor and their destination floor to the set of required floors.
           - For each unserved passenger currently boarded, add their destination floor to the set of required floors.
        5. Estimate the lift movement cost:
           - If there are no required floors, the movement cost is 0.
           - If there are required floors, find the minimum and maximum floor indices among them.
           - Get the index of the current lift floor.
           - The estimated movement cost is calculated as the minimum distance from the current floor to either the minimum or maximum required floor index, plus the distance between the minimum and maximum required floor indices. This estimates the cost to reach one end of the required floor range and traverse the range.
        6. The total heuristic value is the sum of the discrete actions needed
           and the estimated lift movement cost.
        """
        current_lift_floor_name = None
        passengers_waiting_list = [] # List of (p, f)
        passengers_boarded_set = set() # Set of p
        passengers_served_set = set() # Set of p

        # Parse dynamic facts from the state
        for fact_str in state:
            # Fact format: '(predicate arg1 arg2 ...)'
            # Extract predicate and arguments by splitting the string inside the brackets
            parts = fact_str[1:-1].split(' ')
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'lift-at':
                current_lift_floor_name = args[0]
            elif predicate == 'origin':
                p, f = args
                passengers_waiting_list.append((p, f))
            elif predicate == 'boarded':
                p = args[0]
                passengers_boarded_set.add(p)
            elif predicate == 'served':
                p = args[0]
                passengers_served_set.add(p)

        # Check if goal is reached (all passengers served)
        if len(passengers_served_set) == len(self.all_passengers):
             return 0 # Goal state

        # Calculate heuristic components
        actions_needed = 0
        required_floors_names = set()

        unserved_passengers = self.all_passengers - passengers_served_set

        # Identify actions needed and required floors for unserved passengers
        waiting_passengers_dict = dict(passengers_waiting_list) # Make lookup easier

        for p in unserved_passengers:
            if p in passengers_boarded_set:
                # Passenger is boarded, needs to depart at destination
                actions_needed += 1 # Needs depart action
                required_floors_names.add(self.passenger_destinations[p])
            elif p in waiting_passengers_dict:
                # Passenger is waiting at origin, needs to board and then depart
                origin_floor = waiting_passengers_dict[p]
                actions_needed += 1 # Needs board action
                required_floors_names.add(origin_floor)
                required_floors_names.add(self.passenger_destinations[p])
            # else: passenger is unserved but not waiting or boarded? (Should not happen in valid states)


        lift_movement = 0
        if required_floors_names:
            # Convert required floor names to indices, filtering out any unknown floors
            required_floor_indices = {self.floor_to_index[f] for f in required_floors_names if f in self.floor_to_index}

            if required_floor_indices: # Check if any required floors were valid and mapped
                min_req_index = min(required_floor_indices)
                max_req_index = max(required_floor_indices)

                # Get the index of the current lift floor
                # Assuming current_lift_floor_name is always present and in floor_to_index
                if current_lift_floor_name in self.floor_to_index:
                    current_floor_index = self.floor_to_index[current_lift_floor_name]

                    # Estimate movement cost to visit all required floors
                    # This is the distance to reach one extreme of the required range
                    # and then traverse the entire range.
                    dist_to_min = abs(current_floor_index - min_req_index)
                    dist_to_max = abs(current_floor_index - max_req_index)
                    range_dist = max_req_index - min_req_index

                    lift_movement = min(dist_to_min, dist_to_max) + range_dist
                # else: current lift floor name not found in floor map. Assume 0 movement or handle error.
                # For this heuristic, we assume valid states and floor names.

        return actions_needed + lift_movement

