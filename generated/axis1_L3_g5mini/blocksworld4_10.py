from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split into components."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the `transport` (secret_domain) domain.
    The heuristic estimates the number of actions required to satisfy the goal
    facts by reasoning about the minimal number of operator applications
    necessary to establish missing facts of the predicates:
      - (at ?o ?loc)
      - (road ?o)
      - (capacity-predecessor ?o)
      - (capacity)
    It performs a small, bounded search over the relevant boolean fluents
    for each object that appears in the goals to compute a compact estimate
    of how many actions are needed to satisfy all goal atoms concerning that
    object. The heuristic is allowed to be optimistic (non-admissible) and
    focuses on being informative while remaining efficiently computable.

    Assumptions
    -----------
    - The domain's operators are exactly those in the PDDL domain file:
      pick-up, drop, drive, drive-2. The heuristic encodes their preconditions
      and add/delete effects as boolean transitions for the purpose of a
      compact per-object search.
    - Goal atoms are of the predicates: at, road, capacity-predecessor, capacity.
      (This matches the provided domain and example instances.)
    - We approximate interactions between objects by including only the
      single "target" location involved in an (at ?o ?loc) goal (called road_y).
      This is sufficient to capture common patterns (e.g., drop requires
      road at the target) while keeping the per-object search small.
    - The heuristic ignores other cross-object resource-contention effects
      beyond the capacity boolean variable. This keeps the heuristic fast
      while capturing essential resource dependencies.

    Heuristic Initialization
    ------------------------
    - The constructor stores the set of goal facts and the static facts.
    - Goals are partitioned by predicate into:
      * goal_at: mapping from object -> target location
      * goal_road: set of objects that must have (road ?o)
      * goal_cap_pred: set of objects that must have (capacity-predecessor ?o)
      * goal_capacity: boolean indicating whether (capacity) is required
    - No other static facts are required for this heuristic.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given planner node (state):
    1. If the state's facts already satisfy all goals, return 0.
    2. Parse the state's facts into quick membership booleans:
       - For a particular object x and a target y (if any), we consider:
         in_x, road_x, cap_pred_x, capacity, at_x_y, road_y
       - These booleans are read from the explicit state representation.
    3. For each object that appears in any of the goals (at, road,
       capacity-predecessor), perform a small breadth-first search over the
       boolean state-space formed by the six booleans listed above.
       - The search applies the domain operators (pick-up, drop, drive,
         drive-2) but only instantiated with the current object (and its
         goal target when applicable). Preconditions are checked using the
         booleans described above and effects update those booleans.
       - The BFS looks for the minimal number of operator applications
         required to make all goal facts for that particular object true
         (e.g., make at(x,y) true and/or road(x) true and/or capacity-predecessor(x) true).
       - The BFS is bounded by the finite boolean state-space (at most 64
         states for the 6 booleans) and therefore is efficient.
    4. Sum the minimal costs across all goal-relevant objects. Finally,
       if the global (capacity) goal is required and not already satisfied
       by either the initial state or by any per-object plan, add a small
       cost (1) as a fallback to ensure the heuristic recognizes that an
       action will be needed to produce capacity.
    5. Return the integer sum as the heuristic estimate.

    Rationale
    ---------
    - The per-object small BFS captures useful interactions between the
      predicates manipulated by the operators (especially that single
      operators like drop and pick-up achieve multiple fluents simultaneously).
    - The approach avoids heavy lifting (no global search) but is considerably
      more informative than a raw count of missing atoms because it respects
      operator add-effect structure and the key resource predicate `capacity`.
    - The heuristic is finite for solvable states (the per-object boolean
      search either finds a solution in the bounded boolean space or falls
      back gracefully), and returns 0 exactly on states satisfying all goals.
    """

    def __init__(self, task):
        # Store the goal set and static facts.
        self.goals = task.goals
        self.static = task.static

        # Partition goals by predicate
        self.goal_at = {}  # object -> desired location
        self.goal_road = set()
        self.goal_cap_pred = set()
        self.goal_capacity = False

        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "at":
                obj, loc = parts[1], parts[2]
                self.goal_at[obj] = loc
            elif pred == "road":
                self.goal_road.add(parts[1])
            elif pred == "capacity-predecessor":
                self.goal_cap_pred.add(parts[1])
            elif pred == "capacity":
                self.goal_capacity = True

    def __call__(self, node):
        state = node.state

        # 1) If the state already satisfies all goals -> heuristic 0
        if self.goals <= state:
            return 0

        # helpers to test presence of facts in current state
        def has_fact(pred, *args):
            return f"({pred} {' '.join(args)})" in state

        # Build set of objects that need consideration:
        objs = set(self.goal_road) | set(self.goal_cap_pred) | set(self.goal_at.keys())

        total_cost = 0

        # For each object that appears in the goals, run a small bounded BFS
        # over the boolean fluents relevant to that object and its target.
        for obj in objs:
            target = self.goal_at.get(obj, None)  # may be None

            # Initial boolean values for the local simulation
            in_x = has_fact("in", obj)
            road_x = has_fact("road", obj)
            cap_pred_x = has_fact("capacity-predecessor", obj)
            capacity = has_fact("capacity")
            at_x_y = False
            road_y = False
            if target is not None:
                at_x_y = has_fact("at", obj, target)
                road_y = has_fact("road", target)

            # Determine which of the goals for this object need to be satisfied
            need_at = (target is not None) and (not at_x_y)
            need_road = (obj in self.goal_road) and (not road_x)
            need_cap_pred = (obj in self.goal_cap_pred) and (not cap_pred_x)

            # If no goals for this object are missing, cost 0 for this object.
            if not (need_at or need_road or need_cap_pred):
                continue

            # BFS over the small boolean state-space: (in_x, road_x, cap_pred_x, capacity, at_x_y, road_y)
            start = (in_x, road_x, cap_pred_x, capacity, at_x_y, road_y)
            # Define goal-check function for this object's required goal set
            def goal_satisfied(state_tuple):
                in_t, road_t, cappred_t, cap_t, at_t, roady_t = state_tuple
                if need_at and not at_t:
                    return False
                if need_road and not road_t:
                    return False
                if need_cap_pred and not cappred_t:
                    return False
                return True

            # BFS
            visited = set()
            q = deque()
            q.append((start, 0))
            visited.add(start)
            found_cost = None
            # limit is implicit by the finite boolean space; we still avoid infinite loops
            while q:
                cur, depth = q.popleft()
                if goal_satisfied(cur):
                    found_cost = depth
                    break
                in_t, road_t, cappred_t, cap_t, at_t, roady_t = cur

                # For readability: possible actions instantiated for this object and its target

                # Action: pick-up(obj)
                # precondition: in_x
                # effect: road_x = True, capacity = True, cap_pred_x = True, in_x = False
                if in_t:
                    nxt = (False, True, True, True, at_t, roady_t)
                    if nxt not in visited:
                        visited.add(nxt)
                        q.append((nxt, depth + 1))

                # Action: drop(obj, target) -- only meaningful if target exists
                # precondition: in_x and road_y
                # effect: capacity = True, road_x = True, at_x_y = True, road_y = False, in_x = False
                if target is not None and in_t and roady_t:
                    nxt = (False, True, cappred_t, True, True, False)
                    if nxt not in visited:
                        visited.add(nxt)
                        q.append((nxt, depth + 1))

                # Action: drive(obj)
                # precondition: road_x and cap_pred_x and capacity
                # effect: in_x = True, road_x = False, cap_pred_x = False, capacity = False
                if road_t and cappred_t and cap_t:
                    nxt = (True, False, False, False, at_t, roady_t)
                    if nxt not in visited:
                        visited.add(nxt)
                        q.append((nxt, depth + 1))

                # Action: drive-2(obj, target) -- meaningful if target exists
                # precondition: at_x_y and road_x and capacity
                # effect: in_x = True, road_y = True, at_x_y = False, road_x = False, capacity = False
                if target is not None and at_t and road_t and cap_t:
                    nxt = (True, False, cappred_t, False, False, True)
                    if nxt not in visited:
                        visited.add(nxt)
                        q.append((nxt, depth + 1))

            # If BFS found a solution, use that depth as the minimal cost for this object.
            if found_cost is not None:
                total_cost += found_cost
            else:
                # As a fallback (shouldn't normally happen because the boolean space is small),
                # assign a conservative estimate: count each missing predicate as 1 or 2.
                fallback = 0
                if need_at:
                    fallback += 2
                if need_road:
                    fallback += 1
                if need_cap_pred:
                    fallback += 1
                total_cost += fallback

        # Finally, if the global (capacity) goal is required and not present in the state,
        # we may need an extra action to establish it. If some per-object plan already
        # made capacity true, we cannot detect that globally here, so we conservatively
        # add 1 if it's missing.
        if self.goal_capacity and not has_fact("capacity"):
            total_cost += 1

        # Ensure heuristic is an integer and finite
        return int(total_cost)