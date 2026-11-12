from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate obj1 obj2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (aixg4uih).

    Summary:
    This heuristic estimates the minimal number of actions required to achieve the goal
    conditions, which are of the form (qxafq0yb <z65nyce9-object>). The key goal predicate is qxafq0yb.
    The heuristic focuses on tracking which goal objects have achieved the predicate qxafq0yb, and
    which still need to achieve it. 

    Key domain dynamics:
    - The goal requires certain z65nyce9-typed objects to satisfy (qxafq0yb ?).
    - z65nyce9 objects gain qxafq0yb primarily when certain actions are applied that require:
       * The object to have (zmomsdb3 ?) and (waexgd4o ? ?imdah)
       * Another predicate (dym1mop6 ?c3xfw ?axwhj) to be falseed
       * Complex conditions with a ggld6jmz object, s1clodkz-typed object, and u59rqa6u constants.
    - The predicate (qrohx31m <s1clodkz> <u59rqa6u>) is mutable and changeable by ss17onrb actions.
    - The predicate (dym1mop6 ?) indicates some "active" link; it can be created by nzh0a058, and removed by l6w9j7tw or eav722gv.
    - Actions affecting qxafq0yb involve changes on objects of various types coordinated via these predicates.

    Assumptions:
    - To achieve (qxafq0yb O), the object O must have (zmomsdb3 O) true.
    - The predicate (qxafq0yb O) is produced by actions affecting (dym1mop6 ?c3xfw ?axwhj) and corresponding related predicates.
    - (qrohx31m ?axwhj ?imdah) can be "moved" from one ?imdah to another via ss17onrb actions; this allows enabling actions that delete (dym1mop6).
    - The constants s154x56h and the unique u59rqa6u objects like s154x56h are special and serve as key connectors in the domain.
    - The heuristic does not attempt full solution but measures how many goal qxafq0yb predicates remain unachieved,
      and roughly estimates number of steps needed to activate or move required predicates based on static facts.

    Heuristic Initialization:
    - Extract all goal objects for which (qxafq0yb ?) must hold.
    - Extract static predicates for quick queries such as (zmomsdb3 ?), (waexgd4o ? ?imdah),
      and the unique constant s154x56h of type u59rqa6u.
    - Extract all s1clodkz objects and ggld6jmz objects appearing in static facts or init for reference.

    Step-By-Step Thinking for Computing Heuristic:
    1. Determine which goal z65nyce9 objects have already achieved (qxafq0yb ?).
    2. Count how many goal objects are not yet achieved (missing qxafq0yb).
    3. For each missing goal object O:
       - Check if (zmomsdb3 O) holds in static (as it is static).
       - Check which u59rqa6u object ?imdah is currently associated with O via (waexgd4o O ?imdah) from static.
       - Check if the needed (qrohx31m ?axwhj ?imdah) holds for some ?axwhj (s1clodkz), else estimate ss17onrb actions needed.
       - If (dym1mop6 ?c3xfw ?axwhj) holds for some ggld6jmz and s1clodkz, then only one final action (l6w9j7tw or eav722gv) is needed to achieve qxafq0yb.
       - Otherwise, a more complex sequence of actions is needed: possibly a nzh0a058 to create (dym1mop6), ss17onrb to move (qrohx31m), then l6w9j7tw/eav722gv to achieve qxafq0yb.
    4. We sum estimated minimal counts of required actions over all missing goal objects.
    5. To keep computation efficient, we only count 0 if goal achieved,
       and otherwise add a fixed action cost per missing qxafq0yb plus an extra cost per missing (dym1mop6) or (qrohx31m)-position.

    The heuristic value is therefore roughly:
    - 0 if all goal objects have qxafq0yb,
    - otherwise: (#missing qxafq0yb) * base cost + (extra costs for fixing missing dym1mop6 or qrohx31m).

    This heuristic is not admissible but guides the planner effectively by focusing on progress towards each goal qxafq0yb predicate.
    """

    def __init__(self, task):
        """Initialize heuristic by extracting goal objects, static facts, and constants."""

        self.goals = task.goals  # goal facts (qxafq0yb ? objects)
        self.static = task.static  # static facts unchanged by actions

        # Extract all goal objects O for which (qxafq0yb O) must hold
        self.goal_objects = set()
        for fact in self.goals:
            pred, *args = get_parts(fact)
            if pred == "qxafq0yb" and len(args) == 1:
                self.goal_objects.add(args[0])

        # Extract (zmomsdb3 ?) for quick lookup
        self.zmomsdb3_set = {
            get_parts(fact)[1] for fact in self.static if match(fact, "zmomsdb3", "*")
        }

        # Map each z65nyce9 object O to its associated u59rqa6u object ?imdah via (waexgd4o O ?imdah)
        self.waexgd4o_map = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "waexgd4o" and len(parts) == 3:
                obj_z65, obj_u59 = parts[1], parts[2]
                self.waexgd4o_map[obj_z65] = obj_u59

        # Extract all (qrohx31m ?axwhj ?imdah) static facts (if any)
        # Usually (qrohx31m ?) are not static but we'll keep track if any appear static.
        self.qrohx31m_static = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "qrohx31m" and len(parts) == 3:
                self.qrohx31m_static.add((parts[1], parts[2]))

        # Extract the unique constant s154x56h of type u59rqa6u
        # From domain we know s154x56h is a constant in u59rqa6u
        self.special_u59rqa6u_const = "s154x56h"

    def __call__(self, node):
        """Estimate the minimal number of actions required to achieve all goals from the current state."""

        state = node.state

        # Quick goal test: if all goal (qxafq0yb ?) present => heuristic 0
        goal_achieved = True
        current_qxafq0yb_objs = set()
        for fact in state:
            if match(fact, "qxafq0yb", "*"):
                current_qxafq0yb_objs.add(get_parts(fact)[1])
        if self.goal_objects <= current_qxafq0yb_objs:
            return 0  # goal reached

        missing_objects = self.goal_objects - current_qxafq0yb_objs
        goal_achieved = False

        # Helper sets for checking presence of key predicates
        dym1mop6_facts = set()
        qrohx31m_facts = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "dym1mop6" and len(parts) == 3:
                dym1mop6_facts.add((parts[1], parts[2]))
            if parts[0] == "qrohx31m" and len(parts) == 3:
                qrohx31m_facts.add((parts[1], parts[2]))

        xx5ljjai_set = {
            get_parts(fact)[1] for fact in state if match(fact, "xx5ljjai", "*")
        }
        omcsgizs_set = {
            get_parts(fact)[1] for fact in state if match(fact, "omcsgizs", "*")
        }
        okcl3dr0_set = {
            get_parts(fact)[1] for fact in state if match(fact, "okcl3dr0", "*")
        }  # some ggld6jmz predicate useful in some actions

        # Heuristic cost accumulator
        total_cost = 0

        # Constants for cost estimates (tuned roughly, as heuristic need not be admissible)
        COST_BASE_PER_MISSING_QXAFQ0YB = 4  # cost to establish qxafq0yb for one object, roughly
        COST_DYM1MOP6_MISSING = 2           # extra cost if dym1mop6 link is missing and must be created
        COST_QROHX31M_FIX = 2              # cost for adjusting the qrohx31m predicate with ss17onrb action(s)

        # For each missing goal object, estimate action steps needed
        for obj in missing_objects:
            cost_for_obj = 0

            # Check if zmomsdb3 condition holds for the object (should be static)
            if obj not in self.zmomsdb3_set:
                # If missing, add a penalty (likely unsolvable or expensive to fix)
                cost_for_obj += 6  # large penalty, as this must hold for qxafq0yb to be achievable

            # Find associated u59rqa6u object via waexgd4o static relation
            imdah = self.waexgd4o_map.get(obj, None)

            # If no association, it may be very hard or impossible to reach qxafq0yb
            if imdah is None:
                cost_for_obj += 6
                total_cost += cost_for_obj
                continue  # skip further detailed checks for this object

            # The qrohx31m predicate (s1clodkz, u59rqa6u) must hold to enable relevant actions
            # We look if some s1clodkz axwhj satisfies (qrohx31m axwhj imdah)
            qrohx31m_known_axwhj = [
                axwhj for (axwhj, ximdah) in qrohx31m_facts if ximdah == imdah
            ]
            if not qrohx31m_known_axwhj:
                # Try to check if static or dynamic qrohx31m facts exist for the pair
                static_match = any(imdah == ximdah for (_, ximdah) in self.qrohx31m_static)
                if not static_match:
                    # Need to move or create qrohx31m predicate using ss17onrb actions
                    cost_for_obj += COST_QROHX31M_FIX

            # Now check if a dym1mop6 fact exists for some ggld6jmz and s1clodkz axwhj linked to imdah
            # dym1mop6 ?c3xfw ?axwhj with (qrohx31m ?axwhj ?imdah) is needed to apply l6w9j7tw/eav722gv 
            # to get qxafq0yb for obj.

            # We try to detect at least one dym1mop6 with matching axwhj linked to imdah via qrohx31m_facts
            found_dym1mop6_for_obj = False
            for (c3xfw, axwhj) in dym1mop6_facts:
                if (axwhj, imdah) in qrohx31m_facts:
                    found_dym1mop6_for_obj = True
                    break
            if not found_dym1mop6_for_obj:
                # This means we need an extra action to create dym1mop6 (nzh0a058)
                cost_for_obj += COST_DYM1MOP6_MISSING

            # Base cost for achieving qxafq0yb after predicates align
            cost_for_obj += COST_BASE_PER_MISSING_QXAFQ0YB

            total_cost += cost_for_obj

        # Ensure heuristic finite and non-zero for non-goal states
        if total_cost == 0 and not goal_achieved:
            total_cost = 1

        return total_cost
