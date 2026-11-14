from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract parts of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches the given pattern with wildcards.

    Args:
        fact (str): A fact string, e.g. "(kut2ksnk p1 f6)".
        *args: Pattern parts, e.g. ("kut2ksnk", "*", "f6").

    Returns:
        bool: True if fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (ud3uf11z).

    Summary
    -------
    This heuristic estimates the number of actions required to reach a goal state
    where each object of type nxbd1hc3 satisfies the predicate (a7sb56om ?object).
    The domain describes a process where objects "praz5" can be in certain "omggbn9z"-marked
    locations and have two possible binary states represented by facts f2xo7hb6 and
    a7sb56om. The goal is that the target objects have a7sb56om true.

    Assumptions
    -----------
    - A goal is reached when all goal (a7sb56om ?p) facts hold.
    - Objects start with either kut2ksnk or klw11kjf relations to certain location objects.
    - Locations with omggbn9z are "active" or current locations where actions can occur.
    - Objects need to have their predicates transitioned from kut2ksnk/klw11kjf + omggbn9z, 
      through setting f2xo7hb6, then switching to a7sb56om.
    - Locations can be "moved" via acovodrf relations from one to another by actions qs8pdrqj or i0veo5kv,
      allowing changing the current omggbn9z location.
    - Actions diwlmgg9 and y7nik9q2 allow transitioning an object per location.
    - The main challenge is to move omggbn9z predicates to the right location(s) for objects
      to progress their state and finally get a7sb56om.

    Heuristic Initialization
    ------------------------
    - Extract from the static facts the acovodrf adjacency graph between location objects;
      this graph is used to estimate shortest distance (in location "moves") between any two locations.
    - Identify all nxbd1hc3 objects that appear in goal conditions (to reach a7sb56om).
    - Extract static kut2ksnk and klw11kjf relations from static facts to find their fixed locations,
      as these do not change according to domain definition.
    - Store initial omggbn9z location(s) as known from the current state.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify all goal target objects (nxbd1hc3) that require (a7sb56om ?obj).
       For these objects:
       - If they already have a7sb56om in the current state, no further cost is needed for them.
    2. For each object which does not have a7sb56om:
       - If the object has f2xo7hb6 true, then the last action y7nik9q2 must be applied at the location
         where (klw11kjf ?obj ?loc) holds and omggbn9z is at that location to produce a7sb56om.
         So estimate cost as:
            cost_to_move_omggbn9z_to_klw11kjf_location + 1 (for y7nik9q2)
       - Otherwise (no f2xo7hb6):
         The object must first apply diwlmgg9 at a location where (kut2ksnk ?obj ?loc) holds and omggbn9z is there,
         costing 1 for diwlmgg9 plus cost to move omggbn9z there if needed.
         Then, it must move omggbn9z to (klw11kjf ?obj ?loc) location and apply y7nik9q2 (1 action).
         Therefore, cost:
           cost_to_move_omggbn9z_to_kut2ksnk_location + 1 (diwlmgg9) +
           cost_to_move_omggbn9z_from_kut2ksnk_to_klw11kjf_location +
           1 (y7nik9q2)
    3. Movement cost of omggbn9z is counted as distance between locations in the acovodrf adjacency graph.
       Each move costs 1 (actions qs8pdrqj or i0veo5kv).
    4. The total heuristic is the sum over all target objects of their minimum estimated costs.
    5. For efficiency, precompute shortest paths between all locations using BFS on acovodrf graph.
    6. Use current omggbn9z location(s) from state and assume we can move omggbn9z arbitrarily between locations.
       If multiple omggbn9z facts exist, pick minimal distances accordingly.
    7. Only count the minimal actions as above; do not enforce admissibility, but keep it informative.
    """

    def __init__(self, task):
        # Goal facts requiring (a7sb56om ?p)
        self.goal_objects = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "a7sb56om" and len(parts) == 2:
                self.goal_objects.add(parts[1])

        # Extract static facts
        self.kut2ksnk = {}  # object -> location
        self.klw11kjf = {}  # object -> location
        self.adjacency = {}  # location -> set of neighbor locations

        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "kut2ksnk" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                self.kut2ksnk[obj] = loc
            elif pred == "klw11kjf" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                self.klw11kjf[obj] = loc
            elif pred == "acovodrf" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.adjacency.setdefault(loc1, set()).add(loc2)
                # Note: according to domain, acovodrf links seem directed,
                # but two actions move forward/backward - so consider undirected graph:
                self.adjacency.setdefault(loc2, set()).add(loc1)

        # Precompute shortest distances between all locations using BFS:
        self.location_distances = {}
        self.locations = list(self.adjacency.keys())

        for loc in self.locations:
            self.location_distances[loc] = self._bfs_distances_from(loc)

    def _bfs_distances_from(self, start):
        """Compute shortest distances from start location to all reachable locations."""
        from collections import deque

        queue = deque([(start, 0)])
        distances = {start: 0}
        while queue:
            curr, dist = queue.popleft()
            for nbr in self.adjacency.get(curr, ()):
                if nbr not in distances:
                    distances[nbr] = dist + 1
                    queue.append((nbr, dist + 1))
        return distances

    def _min_dist_omggbn9z_to(self, omggbn9z_locs, target_loc):
        """
        Compute the minimal number of move actions to get omggbn9z from any of the current locations
        to the target location. Return 0 if any omggbn9z is already at target_loc.
        """
        min_dist = None
        for loc in omggbn9z_locs:
            dist = self.location_distances.get(loc, {}).get(target_loc)
            if dist is not None:
                if min_dist is None or dist < min_dist:
                    min_dist = dist
            elif loc == target_loc:
                # already at target location
                return 0
        return min_dist if min_dist is not None else 9999  # Large if unreachable

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.

        For each goal object (must have a7sb56om), estimate cost to achieve it:
         - If already a7sb56om, cost 0.
         - If f2xo7hb6 is true:
           cost = minimal move omggbn9z to klw11kjf location + 1 (apply y7nik9q2)
         - Else:
           cost = minimal move omggbn9z to kut2ksnk location + 1 (diwlmgg9)
                  + move from kut2ksnk loc to klw11kjf loc
                  + 1 (y7nik9q2)

        Sum over all goal objects.

        Returns:
            int: Estimated cost to reach the goal from the current state.
        """
        state = node.state

        # If goal already satisfied, heuristic is zero
        if self.goal_objects and all(
            f"(a7sb56om {obj})" in state for obj in self.goal_objects
        ):
            return 0

        # Current locations of omggbn9z predicate
        current_omggbn9z_locs = {
            get_parts(fact)[1] for fact in state if fact.startswith("(omggbn9z ")
        }
        if not current_omggbn9z_locs:
            # No omggbn9z locations known, heuristic cannot progress; assign large cost:
            return 9999

        total_cost = 0

        for obj in self.goal_objects:
            goal_fact = f"(a7sb56om {obj})"
            if goal_fact in state:
                # Already achieved for this object
                continue

            f2xo7hb6_fact = f"(f2xo7hb6 {obj})"
            # Locations from static relations (kut2ksnk and klw11kjf must exist for each obj)
            kut2_loc = self.kut2ksnk.get(obj)
            klw1_loc = self.klw11kjf.get(obj)

            if klw1_loc is None or kut2_loc is None:
                # Missing static location info is problematic; fallback large cost
                total_cost += 9999
                continue

            if f2xo7hb6_fact in state:
                # Already in intermediate f2xo7hb6 state, need to apply y7nik9q2 at klw11kjf loc.
                dist_to_klw1 = self._min_dist_omggbn9z_to(current_omggbn9z_locs, klw1_loc)
                if dist_to_klw1 is None:
                    dist_to_klw1 = 9999
                # 1 action for y7nik9q2
                obj_cost = dist_to_klw1 + 1
                total_cost += obj_cost
            else:
                # Need first diwlmgg9 at kut2ksnk location then y7nik9q2 at klw11kjf location
                dist_to_kut2 = self._min_dist_omggbn9z_to(current_omggbn9z_locs, kut2_loc)
                if dist_to_kut2 is None:
                    dist_to_kut2 = 9999
                dist_between_locs = self.location_distances.get(kut2_loc, {}).get(klw1_loc)
                if dist_between_locs is None:
                    dist_between_locs = 9999

                # Actions: diwlmgg9 and y7nik9q2 each cost 1
                obj_cost = dist_to_kut2 + 1 + dist_between_locs + 1
                total_cost += obj_cost

        # Defensive: heuristic cost should never be zero unless goal is reached
        if total_cost == 0:
            return 1

        return total_cost
