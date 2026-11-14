from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Remove surrounding parentheses from a fact string and split into parts.
    Example: "(at loc1 loc2)" -> ["at", "loc1", "loc2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A simple, domain-dependent heuristic for the `transport` (secret_domain) domain.
    The heuristic estimates how many actions are required to reach the goal by
    counting unsatisfied goal facts and estimating a small sequence of actions
    needed to achieve each missing fact. It is not admissible but is cheap to
    compute and discriminative for guiding GBFS in this domain.

    Assumptions
    -----------
    - Goals are conjunctions of ground predicates of the forms:
        (at OBJECT LOCATION), (road OBJECT), (capacity-predecessor OBJECT), and/or (capacity)
    - The planner state uses only the predicates defined in the domain:
        road, capacity-predecessor, capacity, in, at
    - We do not attempt to compute exact minimal action sequences; rather,
      we estimate a small number of actions that are typically required to
      establish a missing goal fact. This estimation trades off exactness
      for speed and robustness.
    - All objects referenced in facts are represented as simple tokens (no
      extra parentheses inside tokens), so splitting on whitespace after
      trimming parentheses suffices.

    Heuristic Initialization
    ------------------------
    - The constructor stores the set of goal facts and the static facts
      (even if unused for these particular PDDL files).
    - It also preprocesses goal facts to categorize them into:
        * goal_at: a mapping object -> desired successor/location
        * goal_road: set of objects that must have (road OBJECT)
        * goal_capacity_predecessor: set of objects that must have (capacity-predecessor OBJECT)
        * goal_capacity: boolean whether (capacity) must hold in the goal

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given node/state:
    1. If the state's facts already contain all goal facts, return 0.
    2. Parse the current state into quick lookup structures:
       - at_map: object -> current location (from "(at obj loc)")
       - in_set: objects currently carried (from "(in obj)")
       - road_set: objects that currently have (road obj)
       - cap_pred_set: objects that currently have (capacity-predecessor obj)
       - capacity_present: whether "(capacity)" is true in the state
    3. Initialize an integer cost = 0.
    4. For each missing goal fact, add a small, finite estimated cost:
       - Goal (at O T) missing:
         * If O is currently carried (in_set):
             - If target T already has road(T) then a single drop typically suffices,
               so add cost 1.
             - Otherwise, creating road(T) and then dropping usually takes an extra step,
               so add cost 2.
         * If O is not carried:
             - We estimate the cost to make O "in" (1 if the local conditions look favorable,
               else 2), and then the cost to drop onto T (1 if road(T) is present else 2).
             - Concretely: cost_make_in = 1 if (road O and capacity-predecessor O and capacity)
               are present, else 2. Then add cost_make_in + (1 if road T else 2).
       - Goal (road O) missing:
         * If O is currently carried, pick-up can usually make road(O) in one action -> +1.
         * Otherwise, creating road(O) typically requires bringing O into the right state -> +2.
       - Goal (capacity-predecessor O) missing:
         * This predicate is normally produced by pick-up; if O is carried -> +1 else +2.
       - Goal (capacity) missing:
         * Either a pick-up or drop creates capacity -> +1.
    5. Sum all these small estimates and return the integer sum.

    Rationale
    ---------
    - The heuristic tries to capture that achieving an (at ...) goal generally requires
      creating an "in" situation for the transported object and then performing a drop,
      and that creating predicates like road and capacity-predecessor often require
      manipulating the object to be carried first. The numeric estimates are conservative
      but finite and cheap to compute.
    - The heuristic returns 0 exactly when the goal facts are all present, and always
      returns a finite integer for states reachable in this domain.
    """

    def __init__(self, task):
        # store full goal set and static facts
        self.goals = task.goals
        self.static = task.static

        # Preprocess goals into convenient containers
        self.goal_at = {}  # object -> desired location
        self.goal_road = set()
        self.goal_capacity_predecessor = set()
        self.goal_capacity = False

        for g in self.goals:
            parts = get_parts(g)
            pred = parts[0]
            if pred == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                self.goal_at[obj] = loc
            elif pred == "road" and len(parts) >= 2:
                self.goal_road.add(parts[1])
            elif pred == "capacity-predecessor" and len(parts) >= 2:
                self.goal_capacity_predecessor.add(parts[1])
            elif pred == "capacity":
                self.goal_capacity = True

    def __call__(self, node):
        state = node.state

        # If goal already satisfied, heuristic is zero
        if self.goals <= state:
            return 0

        # Parse current state into quick lookup structures
        at_map = {}
        in_set = set()
        road_set = set()
        cap_pred_set = set()
        capacity_present = False

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at" and len(parts) >= 3:
                at_map[parts[1]] = parts[2]
            elif pred == "in" and len(parts) >= 2:
                in_set.add(parts[1])
            elif pred == "road" and len(parts) >= 2:
                road_set.add(parts[1])
            elif pred == "capacity-predecessor" and len(parts) >= 2:
                cap_pred_set.add(parts[1])
            elif pred == "capacity":
                capacity_present = True

        cost = 0

        # Handle (at ...) goal facts
        for obj, desired_loc in self.goal_at.items():
            current_loc = at_map.get(obj)
            # If already at desired location, nothing to do
            if current_loc == desired_loc:
                continue

            # Estimate cost to make (at obj desired_loc) true
            if obj in in_set:
                # object is already carried -> usually just one drop if target has road
                if desired_loc in road_set:
                    cost += 1
                else:
                    # need to create road(desired_loc) and then drop
                    cost += 2
            else:
                # need to make object "in" first
                # If local conditions are favorable (road obj, capacity-predecessor obj, and capacity),
                # a drive-like action may produce in(obj) in one step; otherwise assume 2.
                if obj in road_set and obj in cap_pred_set and capacity_present:
                    cost_make_in = 1
                else:
                    cost_make_in = 2
                # then drop at desired_loc: 1 if target has road, else 2
                cost_drop = 1 if desired_loc in road_set else 2
                cost += cost_make_in + cost_drop

        # Handle (road ...) goal facts missing
        for obj in self.goal_road:
            if obj in road_set:
                continue
            # If object is carried, pick-up can make road(obj) in 1 step; else assume 2
            cost += 1 if obj in in_set else 2

        # Handle (capacity-predecessor ...) goal facts missing
        for obj in self.goal_capacity_predecessor:
            if obj in cap_pred_set:
                continue
            # Usually created by pick-up; cheaper if object is carried
            cost += 1 if obj in in_set else 2

        # Handle (capacity) goal if required
        if self.goal_capacity and not capacity_present:
            # pick-up or drop typically creates capacity -> 1
            cost += 1

        # Ensure we return a finite integer (zero handled earlier)
        return int(cost)