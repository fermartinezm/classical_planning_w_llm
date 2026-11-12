from heuristics.heuristic_base import Heuristic
from collections import deque, defaultdict


def get_parts(fact):
    """Return the parts of a fact string, removing surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the miconic elevator domain.

    The heuristic estimates the number of actions required to serve all
    passengers (i.e., make (served p) true for every passenger p). It
    models the elevator (lift) movement between floors plus the necessary
    boarding and departing (disembark) actions. The heuristic is not
    required to be admissible; instead it aims to be informative and
    cheaply computable to reduce node expansions in greedy best-first search.

    Assumptions
    -----------
    - Each board/depart action costs 1.
    - Each up/down (moving the lift between two adjacent floors)
      action costs 1.
    - The static relation (destin p f) provides each passenger's
      destination floor and is available in the task.static set.
    - The static relation (above f1 f2) defines adjacency between
      floors. We treat this as an undirected adjacency (the lift can
      move between the two floors in one action, either via `up` or `down`)
      since the domain provides both up and down operators using `above`.
    - A passenger is considered "served" if "(served p)" is present
      in the current state.
    - A passenger is considered "boarded" if "(boarded p)" is present.
    - Origin facts "(origin p f)" appear in the current state iff the
      passenger is not boarded yet (board removes the origin fact).
    - The heuristic is finite for connected floor graphs; if floors are
      disconnected, unreachable floors are treated by assigning a large
      finite penalty.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - self.destin: a mapping passenger -> destination_floor (from static facts).
    - self.floors: set of floors encountered in the static `above` facts.
    - self.adj: adjacency dictionary for floors derived from (above ...) facts.
    - We precompute BFS-based shortest-path distances on-demand via a helper
      function using the adjacency graph. Floors that are not connected will
      yield None and be penalized in __call__ by a large finite cost.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all passengers in self.destin are already served (i.e., their
       served fact exists in the state), return 0.
    2. Determine the current lift floor from "(lift-at f)" in the state.
    3. Collect passenger status:
       - served: skip (no cost).
       - boarded: needs to be delivered to its destination (one depart
         action plus the cost to move the lift to that destination).
       - not boarded: has an origin in the current state; it needs a
         board action at its origin, movement to its destination, and a
         depart action.
    4. Use a greedy simulation to estimate a short sequence of actions:
       - Start from the current lift floor.
       - First deliver all currently boarded passengers in nearest-first order:
         repeatedly pick the boarded passenger whose destination is closest to the
         lift, move there (distance cost), and perform depart (+1).
       - Then handle remaining passengers (not boarded). Repeatedly:
         - Move to the nearest origin floor from the lift position (movement cost).
         - Board all passengers whose origin is that floor (+1 per passenger).
         - After boarding, deliver passengers in nearest-first order (move + depart).
       - Continue until all passengers are served.
    5. If any required move is impossible (no path between floors in the
       adjacency graph), a large finite penalty is added for that passenger
       instead of infinity, ensuring heuristic finiteness.
    6. The returned value is the sum of movement steps and all board/depart actions
       as estimated by the greedy simulation. This heuristic may overcount
       actions (not admissible) but captures travel costs and pick/drop costs
       and generally guides search toward states that reduce travel and pending
       passenger tasks.

    The heuristic is intentionally greedy and simulative to capture the benefit
    of serving multiple passengers that share floors or that are on similar
    travel paths. It is efficient (polynomial in the number of passengers and
    floors) and finite for connected floor graphs.
    """

    # Penalty used when floors are disconnected (finite large number).
    DISCONNECTED_PENALTY = 1000

    def __init__(self, task):
        """Extract static information: destin mapping and floor adjacency."""
        static_facts = task.static

        # Map passenger -> destination floor from static "destin" facts.
        self.destin = {}
        # Build adjacency graph for floors from "above" static facts.
        self.adj = defaultdict(set)
        self.floors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "destin" and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.destin[passenger] = floor
                self.floors.add(floor)
            elif pred == "above" and len(parts) == 3:
                f1, f2 = parts[1], parts[2]
                # treat above as undirected adjacency: add both directions
                self.adj[f1].add(f2)
                self.adj[f2].add(f1)
                self.floors.add(f1)
                self.floors.add(f2)

        # For convenience, precompute BFS distances on demand and cache them.
        self._dist_cache = {}

    def _bfs_distance(self, start, goal):
        """Return shortest distance (in lift moves) between start and goal floors."""
        # Same floor -> zero moves
        if start == goal:
            return 0

        key = (start, goal)
        if key in self._dist_cache:
            return self._dist_cache[key]

        if start not in self.adj or goal not in self.adj:
            # If a floor has no adjacency information, treat as possibly isolated.
            self._dist_cache[key] = None
            return None

        visited = {start}
        q = deque([(start, 0)])
        while q:
            cur, d = q.popleft()
            for nb in self.adj[cur]:
                if nb == goal:
                    self._dist_cache[key] = d + 1
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))

        # No path found
        self._dist_cache[key] = None
        return None

    def __call__(self, node):
        """Estimate number of actions from the given state to a goal state."""
        state = node.state

        # Quick goal check: if all passengers with destin are served -> 0
        served_passengers = {get_parts(f)[1] for f in state if get_parts(f)[0] == "served"}
        if all(p in served_passengers for p in self.destin.keys()):
            return 0

        # Find lift current floor
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at":
                lift_floor = parts[1]
                break
        # If no lift position known, pessimistically assume extra cost to reach floors.
        if lift_floor is None:
            # choose any known floor as starting point or bail with penalty
            if self.floors:
                lift_floor = next(iter(self.floors))
            else:
                return 0  # degenerate; nothing to do

        # Determine passenger sets
        boarded = set()     # passengers currently boarded (need delivery)
        not_boarded = {}    # passenger -> origin floor
        all_passengers = set(self.destin.keys())

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "boarded" and len(parts) == 2:
                boarded.add(parts[1])
            elif pred == "origin" and len(parts) == 3:
                p, f = parts[1], parts[2]
                not_boarded[p] = f
            # served facts already handled

        # Filter out already served passengers
        to_serve = {p for p in all_passengers if p not in served_passengers}
        boarded &= to_serve
        not_boarded = {p: f for p, f in not_boarded.items() if p in to_serve}

        cost = 0

        # First deliver currently boarded passengers in nearest-first order.
        # This simulates finishing deliveries for passengers already inside.
        boarded_list = set(boarded)  # mutable set
        current_floor = lift_floor

        while boarded_list:
            # choose passenger with nearest destination
            best_p = None
            best_d = None
            best_dist = None
            for p in boarded_list:
                dest = self.destin.get(p)
                if dest is None:
                    # If no destination info, add a penalty and skip detailed distance
                    dist = None
                else:
                    dist = self._bfs_distance(current_floor, dest)
                if dist is None:
                    # treat as very large distance; still pick it eventually
                    dist_val = self.DISCONNECTED_PENALTY
                else:
                    dist_val = dist
                if best_p is None or dist_val < best_dist:
                    best_p = p
                    best_dist = dist_val
                    best_d = dest
            # move there
            if best_dist == self.DISCONNECTED_PENALTY:
                cost += self.DISCONNECTED_PENALTY
                # still perform depart action
                cost += 1
                # remove passenger
                boarded_list.remove(best_p)
                # do not update current_floor to avoid mixing None floors
            else:
                # add movement cost
                cost += best_dist
                # depart action
                cost += 1
                current_floor = best_d
                boarded_list.remove(best_p)

        # Then handle remaining unboarded passengers.
        # We simulate visiting origins, boarding all passengers at that floor,
        # then delivering nearest-first, repeating until all served.
        # Board cost = 1 per passenger; depart cost counted in delivery loop.
        remaining_unboarded = dict(not_boarded)
        # Maintain a set of passengers currently inside (boarded during the simulation)
        sim_boarded = set()

        while remaining_unboarded or sim_boarded:
            # If there are no sim_boarded passengers, go pick next origin
            if not sim_boarded:
                # choose nearest origin from current_floor
                best_origin = None
                best_p_list = []
                best_dist = None
                for p, origin in remaining_unboarded.items():
                    dist = self._bfs_distance(current_floor, origin)
                    if dist is None:
                        dist_val = self.DISCONNECTED_PENALTY
                    else:
                        dist_val = dist
                    if best_origin is None or dist_val < best_dist:
                        best_origin = origin
                        best_dist = dist_val
                # Move to best_origin
                if best_dist == self.DISCONNECTED_PENALTY:
                    cost += self.DISCONNECTED_PENALTY
                    # board nothing (cannot reach); to avoid infinite loop, consume one unreachable passenger
                    # find a passenger with that origin and add a penalty for its board+depart as well
                    # pick any passenger with that origin
                    picked_p = next((p for p, o in remaining_unboarded.items() if o == best_origin), None)
                    if picked_p is None:
                        # fallback: remove any remaining passenger with a penalty
                        picked_p = next(iter(remaining_unboarded.keys()))
                        remaining_unboarded.pop(picked_p, None)
                        cost += 2  # board + depart (crudely)
                        continue
                    # add board cost and then try to deliver (which will also be penalized)
                    cost += 1  # board
                    remaining_unboarded.pop(picked_p, None)
                    sim_boarded.add(picked_p)
                    continue
                else:
                    cost += best_dist
                    current_floor = best_origin
                    # board all passengers whose origin is current_floor
                    to_board_here = [p for p, o in remaining_unboarded.items() if o == current_floor]
                    for p in to_board_here:
                        cost += 1  # board action
                        sim_boarded.add(p)
                        remaining_unboarded.pop(p, None)

            # Now deliver any sim_boarded passengers in nearest-first order
            if sim_boarded:
                best_p = None
                best_dest = None
                best_dist = None
                for p in list(sim_boarded):
                    dest = self.destin.get(p)
                    if dest is None:
                        dist_val = self.DISCONNECTED_PENALTY
                    else:
                        dist = self._bfs_distance(current_floor, dest)
                        dist_val = dist if dist is not None else self.DISCONNECTED_PENALTY
                    if best_p is None or dist_val < best_dist:
                        best_p = p
                        best_dest = dest
                        best_dist = dist_val
                if best_dist == self.DISCONNECTED_PENALTY:
                    cost += self.DISCONNECTED_PENALTY
                    cost += 1  # depart (still count)
                    sim_boarded.remove(best_p)
                    # do not update current_floor
                else:
                    cost += best_dist  # moves
                    cost += 1          # depart
                    current_floor = best_dest
                    sim_boarded.remove(best_p)

        return cost