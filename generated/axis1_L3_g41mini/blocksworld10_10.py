from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: A string, e.g. "(road locatable1)".
    - `args`: A sequence of patterns (strings), possibly containing wildcards `*`.
    Returns True if all parts matched the corresponding pattern.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the "secret_domain" (transport) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal state 
    by focusing on how to get the right objects to satisfy the goal predicates:
    particularly, achieving (road ?obj), (at ?obj ?loc), and (capacity-predecessor ?obj)
    as specified in the goal. These predicates reflect a state where certain
    locatables are on the road, positioned at specific locatables, and certain 
    locatables are capacity-predecessors.

    The main challenge lies in modeling the "drive", "pick-up", "drop", and "drive-2"
    actions, which toggle predicates like (road ?v), (in ?v), (capacity-predecessor ?v),
    (capacity), and (at ?v ?loc).

    Assumptions
    -----------
    - Each "road" fact indicates an object currently positioned on the road.
    - (in ?v) indicates that the vehicle/object is currently inside capacity (like carrying).
    - (capacity) is a single system-wide flag toggled on/off by pick-up, drop, and drive actions.
    - (capacity-predecessor ?v) flags locatables related to capacity sequencing or road adjacency.
    - The manipulation of road and capacity flags reflects a handover or transport step between locatables.
    - This domain appears to model a mobility or transport mechanism where objects are picked up and dropped 
      along roads or locations.

    Heuristic Initialization
    ------------------------
    - Extract all objects mentioned in the domain.
    - Extract goal predicates and group them by predicate name:
       * road goals = set of objects required to have (road ?obj).
       * at goals = map from object to goal location.
       * capacity-predecessor goals = set of objects required to have (capacity-predecessor ?obj).
    - Although there are no static facts relevant here, we prepare the heuristic for efficient state queries.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state is a goal state, heuristic is zero.
    2. Identify all objects that need to have (road ?obj) in the goal but currently do not.
    3. Identify all objects that need to be at some location (via (at ?obj ?loc)) in the goal but are not currently at that location.
    4. Identify all objects that need (capacity-predecessor ?obj) in the goal but are missing it.
    5. "capacity" and "in" are singleton flags, whose presence helps or restricts moves.

    6. Count for each object the minimal number of action steps to fix each missing predicate:
       - To establish (road ?obj), the object must not only be on the road but also
         must have capacity and capacity-predecessor flags toggled properly.
         The drive and drop actions switch predicates between road/in and capacity states.
       - To fix location discrepancies in (at ?obj ?loc), drop actions can relocate objects.
       - To gain (capacity-predecessor ?obj), pick-up or drop sequences may be necessary.

    7. A rough estimate:
       - For each object missing (road), at minimum 1 action needed (drive or drop).
       - For each object missing (capacity-predecessor), at least 1 action needed (pick-up or drop).
       - For each object misplaced in (at), at least 1 action (drop) plus one action to move road presence.

    8. Additionally, consider the global "capacity" and "in" facts:
       - If capacity is False, a pick-up action might be needed to gain capacity before driving.
       - If capacity is True and (in ?v) holds for some ?v, consider that object is being carried.

    9. To approximate, sum counts for objects missing road and capacity-predecessor plus count mismatches for (at).
       Adjust by 1 for transitions of capacity/in toggles.

    10. This heuristic is not admissible but guides the search greedily by increasing the estimate with unmet goal predicates.

    This heuristic is efficiently computable and leverages the semantic structure of this transport domain.
    """

    def __init__(self, task):
        # Store goal predicates grouped by type for quick reference.
        self.goal_roads = set()
        self.goal_cap_predecessors = set()
        self.goal_ats = dict()  # object -> loc

        for fact in task.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "road" and len(parts) == 2:
                self.goal_roads.add(parts[1])
            elif pred == "capacity-predecessor" and len(parts) == 2:
                self.goal_cap_predecessors.add(parts[1])
            elif pred == "at" and len(parts) == 3:
                self.goal_ats[parts[1]] = parts[2]

        # We keep all objects encountered in the domain facts for iteration
        # Note: Our state facts can be different objects, so for heuristic, we'll recognize by membership.
        self.all_goal_objects = set(self.goal_roads) | set(self.goal_cap_predecessors) | set(self.goal_ats.keys())

    def __call__(self, node):
        state = node.state

        # If goal reached, return 0
        if self.goal_roads <= state and self.goal_cap_predecessors <= state:
            # Also check at facts
            # All goal at predicates must be present in state
            all_at_goals_met = True
            for obj, loc in self.goal_ats.items():
                fact = f"(at {obj} {loc})"
                if fact not in state:
                    all_at_goals_met = False
                    break
            if all_at_goals_met:
                return 0

        # Helpers for fast lookup
        facts_set = state

        # Check global flags
        capacity_present = "(capacity)" in facts_set
        in_objects = {get_parts(f)[1] for f in facts_set if match(f, "in", "*")}

        # Count missing (road ?obj)
        missing_road_objects = [obj for obj in self.goal_roads if f"(road {obj})" not in facts_set]

        # Count missing (capacity-predecessor ?obj)
        missing_cp_objects = [obj for obj in self.goal_cap_predecessors if f"(capacity-predecessor {obj})" not in facts_set]

        # Count misplaced or missing (at ?obj ?loc)
        misplaced_at_objects = []
        for obj, loc in self.goal_ats.items():
            current_at_facts = [f for f in facts_set if match(f, "at", obj, "*")]
            if not current_at_facts:
                misplaced_at_objects.append(obj)
            else:
                # Check if currently at the goal location
                # exact match fact
                if f"(at {obj} {loc})" not in facts_set:
                    misplaced_at_objects.append(obj)

        heuristic_cost = 0

        # Rough cost estimation strategy:
        # To fix each missing road: 1 action minimum, likely drive or drop.
        heuristic_cost += len(missing_road_objects)

        # To fix each missing capacity-predecessor: 1 action minimum, pick-up or drop.
        heuristic_cost += len(missing_cp_objects)

        # To fix location discrepancies (at): at least 1 action per misplaced object.
        heuristic_cost += len(misplaced_at_objects)

        # Adjust cost depending on capacity and in-object presence:
        # If capacity missing but needed by drive or pick-up, add 1 for pick-up to regain capacity.
        # If any object is currently carried (in), might require drop before other actions.
        # Our heuristic is approximate, so:
        if not capacity_present:
            # If goals require objects on road or capacity-predecessor, must regain capacity once
            if missing_road_objects or missing_cp_objects:
                heuristic_cost += 1
        else:
            # If capacity is present, but some object(s) in 'in', we may need at least one drop action to free capacity
            if in_objects:
                heuristic_cost += 1

        # We increase heuristic for complexity imposed by multiple objects.
        # For every two objects that need road or capacity-predecessor, add a small overhead.
        overhead = (len(missing_road_objects) + len(missing_cp_objects)) // 2
        heuristic_cost += overhead

        return heuristic_cost
