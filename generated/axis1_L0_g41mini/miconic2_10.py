from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Summary:
    --------
    This heuristic estimates the number of required actions to serve all passengers
    who have not yet been served, starting from the current state. Serving a passenger
    involves (1) moving the lift to the passenger's origin floor (if needed), (2) boarding
    the passenger, (3) moving the lift to the passenger's destination floor (if needed),
    and (4) departing the passenger. 

    The heuristic sums the minimal costs for all unserved passengers, approximating the
    number of lift moves and the boarding/departing actions needed. It leverages static
    "above" relations to compute distances between floors efficiently.

    Assumptions:
    ------------
    - Each board and depart action requires one step each.
    - Lift moves (up or down) cost one action per floor change.
    - Boarding or departing a passenger can only be done if the lift is at the correct floor.
    - Lift can serve passengers in any order; the heuristic assumes serving passengers independently
      and sums their minimal costs (no combined route optimization).
    - The heuristic is not necessarily admissible but aims to estimate reasonably well the number of steps needed.
    - The lift initially is at exactly one floor in the current state.

    Heuristic Initialization:
    -------------------------
    - Extract the set of floors from static facts using "above" predicates.
    - Build a directed adjacency list of floors based on "above" relations.
    - Compute a shortest path distance matrix between all floors, considering upward and downward moves.
      This allows quick calculation of minimal lift moves between any two floors.
    - Store the destination floor for each passenger from static facts.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1) Identify:
       - The current floor of the lift.
       - Which passengers are already served.
       - Which passengers are boarded (in the lift).
       - The origin and destination floors of all passengers.

    2) For each passenger not yet served:
       - If boarded:
         * Compute the cost as the minimal number of moves from current lift floor to the passenger's destination floor,
           plus 1 action for depart.
       - If not boarded:
         * Compute the cost as minimal moves from current lift floor to origin floor,
           plus 1 action for board,
           plus minimal moves from origin floor to destination floor,
           plus 1 action for depart.

    3) Since the lift position changes after serving each passenger, but exact ordering is unknown,
       approximate by:
       - For all passengers not yet served, compute the cost assuming the lift is currently at its present floor.
       - Sum these costs naively (which might overestimate), or take the maximum cost among passengers as a tighter estimate.
       
       Here, we choose the sum to guide greedy best-first search effectively.

    4) Return the total estimated actions needed to serve all passengers from the current state.

    Note:
    -----
    This heuristic provides a finite, informative estimate for solvable states and gives zero cost only in goal states.
    """

    def __init__(self, task):
        """Initialize the heuristic by preprocessing floor distances and passenger destinations."""
        self.goals = task.goals
        self.static = task.static

        # Extract all floors from static facts (either directly or indirectly from "above" predicates)
        floors = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "above" and len(parts) == 3:
                floors.add(parts[1])
                floors.add(parts[2])
            elif parts[0] == "destin" and len(parts) == 3:
                floors.add(parts[2])
            elif parts[0] == "origin" and len(parts) == 3:
                floors.add(parts[2])
            elif parts[0] == "lift-at" and len(parts) == 2:
                floors.add(parts[1])
        self.floors = sorted(floors)  # Keep sorted for stable ordering

        # Build graph edges for moves: "above" predicate means floor1 is above floor2 (so can go down from floor1 to floor2),
        # but lift can move both up and down.
        # We'll build undirected graph between floors using above relations and reverse.
        adjacency = {floor: set() for floor in self.floors}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "above" and len(parts) == 3:
                f1, f2 = parts[1], parts[2]
                adjacency[f1].add(f2)
                adjacency[f2].add(f1)  # Lift can move up and down

        # Precompute shortest path distances between all pairs of floors using BFS
        self.floor_distances = {}
        for f_start in self.floors:
            distances = {f: float('inf') for f in self.floors}
            distances[f_start] = 0
            queue = [f_start]
            head = 0
            while head < len(queue):
                current = queue[head]
                head += 1
                curr_dist = distances[current]
                for neighbor in adjacency[current]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = curr_dist + 1
                        queue.append(neighbor)
            self.floor_distances[f_start] = distances

        # Extract passenger destinations from static facts
        self.passenger_dest = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "destin" and len(parts) == 3:
                passenger, dest_floor = parts[1], parts[2]
                self.passenger_dest[passenger] = dest_floor

    def __call__(self, node):
        """Estimate the number of required actions to serve all passengers from the current state."""

        state = node.state

        # Quick goal check: if current state is goal, heuristic=0
        if self.goals <= state:
            return 0

        # Parse facts in current state
        boarded = set()
        served = set()
        origin = {}
        lift_floor = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "boarded":
                boarded.add(parts[1])
            elif parts[0] == "served":
                served.add(parts[1])
            elif parts[0] == "origin" and len(parts) == 3:
                origin[parts[1]] = parts[2]
            elif parts[0] == "lift-at":
                lift_floor = parts[1]

        if lift_floor is None:
            # Defensive fallback: if no lift position known, heuristic not computable, return large estimate
            return 100000

        # Identify unserved passengers
        unserved_passengers = set(origin.keys()).union(self.passenger_dest.keys()) - served

        total_cost = 0

        # For each unserved passenger compute estimated cost to serve
        for p in unserved_passengers:
            # Destination must be known for passenger
            dest_floor = self.passenger_dest.get(p, None)
            if dest_floor is None:
                # If no destination info for passenger, skip (or assign zero)
                continue

            # Origins must be known; if not, we assume unknown origin (costly)
            orig_floor = origin.get(p, None)

            if p in boarded:
                # Passenger is onboard:
                # cost = moves lift from current floor to passenger destination + 1 depart action
                dist_to_dest = self.floor_distances[lift_floor].get(dest_floor, float('inf'))
                cost = dist_to_dest + 1
            else:
                # Passenger not onboard yet:
                # cost ~ moves lift to origin + 1 board + moves origin to destination + 1 depart
                if orig_floor is None:
                    # If origin unknown, estimate large cost
                    cost = 10  # arbitrary penalty
                else:
                    dist_to_orig = self.floor_distances[lift_floor].get(orig_floor, float('inf'))
                    dist_orig_to_dest = self.floor_distances[orig_floor].get(dest_floor, float('inf'))
                    cost = dist_to_orig + 1 + dist_orig_to_dest + 1

            # Defensive: costs must be finite and positive
            if cost == float('inf'):
                # Part unreachable, assign high cost but finite
                cost = 1000

            total_cost += cost

        return total_cost
