from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed for the lift to serve all passengers by moving to their origin and destination floors.

    # Assumptions:
    - The lift can move up or down between floors.
    - Each passenger must be boarded at their origin floor and then transported to their destination floor.
    - The lift can serve multiple passengers in a single trip if their origins and destinations are along the same path.

    # Heuristic Initialization
    - Extract the current state of the lift and passengers.
    - Identify the origin and destination floors for each passenger.
    - Determine the furthest origin and destination floors from the lift's current position.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current floor of the lift.
    2. For each passenger, determine if they have been served.
    3. For unserved passengers, calculate the distance from the lift's current floor to their origin and then to their destination.
    4. Determine the furthest origin the lift needs to visit.
    5. Calculate the total distance from the lift's current position to the furthest origin and then to the furthest destination.
    6. Sum the necessary actions for moving to origins, transporting to destinations, and any additional movements.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting relevant information from the task."""
        self.goals = task.goals
        self.static = task.static

        # Extract the current floor of the lift
        self.lift_floor = None
        for fact in task.initial_state:
            if match(fact, "(lift-at *%)"):
                self.lift_floor = fact.split()[-1]

        # Extract origin and destination floors for each passenger
        self.passengers = {}
        for fact in task.initial_state:
            if match(fact, "(origin *% *)"):
                p, f_origin = fact.split()[-2:]
                self.passengers[p] = {'origin': f_origin, 'dest': None, 'boarded': False}
            if match(fact, "(destin *% *)"):
                p, f_dest = fact.split()[-2:]
                if p in self.passengers:
                    self.passengers[p]['dest'] = f_dest
            if match(fact, "(boarded *%)"):
                p = fact.split()[-1]
                if p in self.passengers:
                    self.passengers[p]['boarded'] = True
            if match(fact, "(served *%)"):
                p = fact.split()[-1]
                if p in self.passengers:
                    self.passengers[p]['served'] = True

        # Precompute the static above relationships for distance calculations
        self.above = {}
        for fact in self.static:
            if match(fact, "(above *% *)"):
                f1, f2 = fact.split()[1], fact.split()[2]
                if f1 not in self.above:
                    self.above[f1] = []
                self.above[f1].append(f2)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current lift floor
        lift_floor = self.lift_floor
        for fact in state:
            if match(fact, "(lift-at *%)"):
                lift_floor = fact.split()[-1]

        # Extract current boarded and served status for passengers
        current_passengers = {}
        for fact in state:
            if match(fact, "(boarded *%)"):
                p = fact.split()[-1]
                current_passengers[p] = {'boarded': True, 'served': False}
            if match(fact, "(served *%)"):
                p = fact.split()[-1]
                current_passengers[p] = {'boarded': False, 'served': True}

        # For each passenger, determine if they need to be served
        unserved_passengers = []
        for p, data in self.passengers.items():
            if not current_passengers.get(p, {}).get('served', False):
                unserved_passengers.append(p)

        if not unserved_passengers:
            return 0

        # Calculate the Manhattan distance between floors
        def distance(f1, f2):
            floors = sorted([f1, f2])
            return abs(int(floors[1].split('f')[-1]) - int(floors[0].split('f')[-1]))

        # Determine the furthest origin and destination
        furthest_origin = None
        furthest_dest = None
        max_dist = 0

        for p in unserved_passengers:
            data = self.passengers[p]
            if data['origin'] is not None and data['dest'] is not None:
                d_origin = distance(lift_floor, data['origin'])
                d_dest = distance(data['origin'], data['dest'])
                total_dist = d_origin + d_dest
                if total_dist > max_dist:
                    max_dist = total_dist
                    furthest_origin = data['origin']
                    furthest_dest = data['dest']

        # Calculate the minimal steps needed
        total_actions = 0

        # Move to the furthest origin
        if furthest_origin != lift_floor:
            total_actions += distance(lift_floor, furthest_origin)

        # Move to the furthest destination
        if furthest_dest != furthest_origin:
            total_actions += distance(furthest_origin, furthest_dest)

        # Add actions for boarding and serving each passenger
        total_actions += 2 * len(unserved_passengers)

        return total_actions
