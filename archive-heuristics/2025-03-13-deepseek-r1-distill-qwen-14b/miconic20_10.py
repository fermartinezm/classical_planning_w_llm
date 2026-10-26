from heuristics.heuristic_base import Heuristic

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all passengers from their origin to destination floors.

    # Assumptions:
    - The lift can move up or down between floors.
    - Each passenger must board the lift at their origin and depart at their destination.
    - The heuristic assumes that the lift can serve each passenger in sequence, moving to each origin and then to the destination.

    # Heuristic Initialization
    - Extract the floor hierarchy from the static facts to determine the distance between floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger, check if they are already served. If not, proceed.
    2. For each unserved passenger, determine their origin and destination.
    3. Calculate the distance from the current lift position to the passenger's origin.
    4. Add 1 action for boarding.
    5. Calculate the distance from the origin to the destination.
    6. Add 1 action for departing.
    7. Sum all these actions for all unserved passengers.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting floor hierarchy from static facts."""
        self.static = task.static
        # Build floor hierarchy
        above_graph = {}
        floors = set()
        for fact in self.static:
            if fact.startswith('(above '):
                f1, f2 = fact[7:-1].split()
                above_graph[f1] = f2
                floors.add(f1)
                floors.add(f2)
        # Find top floor (not a destination in any above fact)
        all_destinations = set(above_graph.values())
        top_floor = None
        for f in floors:
            if f not in all_destinations:
                top_floor = f
                break
        # Build the order from top to bottom
        self.floor_order = {}
        current = top_floor
        index = 0
        while current is not None:
            self.floor_order[current] = index
            index += 1
            current = above_graph.get(current)
        # Add any remaining floors (bottom floor)
        for f in floors:
            if f not in self.floor_order:
                self.floor_order[f] = index
                index += 1

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        current_lift = None
        passengers = {}
        for fact in state:
            if fact.startswith('(lift-at '):
                parts = fact[8:-1].split()
                current_lift = parts[0]
            elif fact.startswith('(origin '):
                parts = fact[7:-1].split()
                p, f = parts
                if p not in passengers:
                    passengers[p] = {}
                passengers[p]['origin'] = f
            elif fact.startswith('(destin '):
                parts = fact[7:-1].split()
                p, f = parts
                if p not in passengers:
                    passengers[p] = {}
                passengers[p]['destin'] = f
        total_cost = 0
        for p, data in passengers.items():
            if f'(served {p})' in state:
                continue
            origin = data['origin']
            destin = data['destin']
            # Get the current lift floor
            if current_lift is None:
                continue  # Should not happen as per domain
            # Compute distance from current lift to origin
            if origin not in self.floor_order or current_lift not in self.floor_order:
                continue  # Should not happen as per domain
            dist_lift_to_origin = abs(self.floor_order[current_lift] - self.floor_order[origin])
            # Compute distance from origin to destination
            if destin not in self.floor_order:
                continue  # Should not happen as per domain
            dist_origin_to_destin = abs(self.floor_order[origin] - self.floor_order[destin])
            # Add the actions
            total_cost += dist_lift_to_origin + 1 + dist_origin_to_destin + 1
        return total_cost
