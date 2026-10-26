from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class miconic11Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    based on the current state. It considers the number of passengers who
    are waiting to board, are currently boarded, and the distance the elevator
    needs to travel to serve them.

    # Assumptions
    - Each passenger needs to board, potentially travel to their destination, and depart.
    - The elevator can serve multiple passengers at once.
    - Moving the elevator up or down incurs a cost.

    # Heuristic Initialization
    - Extract the 'above' relationships between floors from the static facts to
      determine the order of floors.
    - Identify all passengers, their origin floors, and destination floors.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the current floor of the lift.
    2.  Identify passengers who are waiting at their origin floors (not boarded).
    3.  Identify passengers who are boarded but not yet served.
    4.  Calculate the cost for boarding all waiting passengers.
    5.  Calculate the cost for moving the lift to the destination floor for each boarded passenger.
    6.  Calculate the cost for each boarded passenger to depart.
    7.  Calculate the cost for moving the lift to the origin floor for each waiting passenger.
    8.  The heuristic value is the sum of these costs.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract floor order from 'above' facts.
        self.floor_order = {}
        for fact in static_facts:
            if "above" in fact:
                parts = fact[1:-1].split()
                f1 = parts[1]
                f2 = parts[2]
                self.floor_order[f1] = f2

        # Extract passenger origins and destinations.
        self.passenger_origins = {}
        self.passenger_destinations = {}
        for fact in static_facts:
            if "destin" in fact:
                parts = fact[1:-1].split()
                passenger = parts[1]
                floor = parts[2]
                self.passenger_destinations[passenger] = floor

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        def match(fact, *args):
            """Check if a PDDL fact matches a given pattern."""
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Get the current lift location.
        lift_location = None
        for fact in state:
            if "lift-at" in fact:
                lift_location = fact[1:-1].split()[1]
                break

        if not lift_location:
            return float('inf')  # If lift location is unknown, return infinity.

        # Identify waiting passengers.
        waiting_passengers = []
        for fact in state:
            if "origin" in fact:
                parts = fact[1:-1].split()
                waiting_passengers.append((parts[1], parts[2]))  # (passenger, floor)

        # Identify boarded passengers.
        boarded_passengers = []
        for fact in state:
            if "boarded" in fact:
                boarded_passengers.append(fact[1:-1].split()[1])  # passenger

        # Identify served passengers
        served_passengers = []
        for fact in state:
            if "served" in fact:
                served_passengers.append(fact[1:-1].split()[1])

        # If all passengers are served, the heuristic is 0.
        all_served = True
        for passenger, _ in waiting_passengers:
            all_served = False
            break
        for passenger in boarded_passengers:
            all_served = False
            break
        if all_served and len(waiting_passengers) == 0 and len(boarded_passengers) == 0:
            return 0

        heuristic_value = 0

        # Cost for boarding waiting passengers.
        heuristic_value += len(waiting_passengers)

        # Cost for moving to the destination floor and departing for boarded passengers.
        for passenger in boarded_passengers:
            destination = self.passenger_destinations[passenger]
            heuristic_value += 1  # Cost to move to destination
            heuristic_value += 1  # Cost to depart

        # Cost for moving to the origin floor for waiting passengers.
        for passenger, origin in waiting_passengers:
            heuristic_value += 1  # Cost to move to origin

        return heuristic_value
