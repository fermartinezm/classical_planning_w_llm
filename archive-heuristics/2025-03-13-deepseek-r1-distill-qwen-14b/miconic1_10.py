from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the total distance the lift needs to travel to each origin and destination, plus the boarding and departing actions.

    # Assumptions:
    - The lift can move directly between any two floors in a single action per floor difference.
    - Each passenger requires the lift to move to their origin, then to their destination.
    - Multiple passengers on the same origin are handled by counting the origin's distance once.

    # Heuristic Initialization
    - Extract the ordered list of floors from the static facts.
    - Create a mapping from each floor to its index in the ordered list.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current position of the lift.
    2. For each passenger, determine their origin and destination if they are not already served.
    3. For each unique origin, calculate the distance from the current lift position to that origin.
    4. For each passenger, calculate the distance from their origin to their destination.
    5. Sum all the origin distances, destination distances, and add two actions per passenger.
    """

    def __init__(self, task):
        # Extract static facts to determine floor order
        static_facts = task.static
        all_floors = set()
        children = {}
        for fact in static_facts:
            if fact.startswith('(above '):
                parts = fact.split()
                f1 = parts[2]
                f2 = parts[3][:-1]
                all_floors.add(f1)
                all_floors.add(f2)
                if f1 not in children:
                    children[f1] = []
                children[f1].append(f2)

        # Find the root (top floor with no parent)
        root = None
        for f in all_floors:
            if f not in children.values():
                root = f
                break

        # Build the ordered list using BFS
        ordered_floors = []
        queue = [root]
        while queue:
            current = queue.pop(0)
            ordered_floors.append(current)
            if current in children:
                queue.extend(children[current])

        self.floors = ordered_floors
        self.floor_index = {f: i for i, f in enumerate(self.floors)}

    def __call__(self, node):
        state = node.state
        # Find current lift position
        current_lift = None
        for fact in state:
            if fact.startswith('(lift-at '):
                current_lift = fact.split()[2][:-1]
                break
        if not current_lift:
            return 0

        # Extract passengers' origin and destination
        passengers = {}
        for fact in state:
            if fact.startswith('(origin '):
                p = fact.split()[1]
                origin = fact.split()[2][:-1]
                passengers[p] = {'origin': origin}
            if fact.startswith('(destin '):
                p = fact.split()[1]
                dest = fact.split()[2][:-1]
                passengers[p]['dest'] = dest

        # Determine unserved passengers
        unserved = []
        for p, data in passengers.items():
            if f'(served {p})' not in state:
                unserved.append(data)

        if not unserved:
            return 0

        # Calculate distances
        unique_origins = set(p['origin'] for p in unserved)
        sum_origin = 0
        for o in unique_origins:
            if o not in self.floor_index:
                continue
            current_idx = self.floor_index[current_lift]
            o_idx = self.floor_index[o]
            sum_origin += abs(current_idx - o_idx)

        sum_dest = 0
        for p in unserved:
            o = p['origin']
            d = p['dest']
            if o not in self.floor_index or d not in self.floor_index:
                continue
            o_idx = self.floor_index[o]
            d_idx = self.floor_index[d]
            sum_dest += abs(o_idx - d_idx)

        total_actions = sum_origin + sum_dest + 2 * len(unserved)
        return total_actions
