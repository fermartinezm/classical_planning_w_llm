from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate obj1 obj2)".
    - `args`: The expected pattern with support for wildcards '*' using fnmatch.
    - Returns True if the fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (biqhd61z) domain.

    Summary:
    This heuristic estimates the number of actions still required to fulfill the goal of
    achieving the predicate (ivo59apj ?db93w) for each relevant object ?db93w of type bciid6ad (called "child" in examples).
    The domain models a multi-step preparation process involving sandwiches (uguffgcw),
    bread pieces (oxxoho4f), contents (hu57dt8m), trays (eoraohfl), and children (bciid6ad).
    The goal predicates (ivo59apj ?db93w) indicate that the specific child's goal state is reached.

    Assumptions:
    - The goal is to achieve (ivo59apj child) for some set of children specified in goal conditions.
    - Each (ivo59apj child) requires that a sequence of predicates/facts be achieved: broadly,
      a sandwich (?yam4s) must be prepared with bread, content, and placed properly on a tray that is associated with a child.
    - The process roughly follows these logical steps reflected by operators:
      1) "Assemble" or "prepare" sandwich (actions kdzjcqwj or h0xkysus) from bread, content, sandwich ingredients.
      2) Place the prepared sandwich onto a tray (jhzk46zy action).
      3) Deliver the sandwich to a child (actions nfarwtet or xtn2vmez).
      4) Swapping trays or tray contents is possible (whgyhnl0 action), but not necessarily always needed.
    - Since the domain is complex with multiple object types and predicates, a simplest heuristic approximation is:
      count how many children still do NOT have the goal fact (ivo59apj child) true in the state,
      multiplied by a heuristic cost approximating the minimal number of actions needed to "serve" one child.

    Heuristic Initialization:
    - Extract all children appearing in goal facts requiring (ivo59apj child).
    - Extract static facts related to children types (n2ij1fah and dr25ifx3), which influence action applicability.
    - Store the constant bupp7mhz (unique zwhbr5cd object) constant used in tray-related predicates and actions.
    - Extract from static and initial facts the associations eyker22y(child, table) for locating children.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify children with unmet goals: those children where (ivo59apj child) is NOT in state.
    2. For each such child, at minimum the following steps must be performed:
       a) Prepare a sandwich from bread and content (requires actions kdzjcqwj or h0xkysus).
       b) Place the prepared sandwich on a tray which is on the child's table (actions jhzk46zy).
       c) Deliver the sandwich to the child (actions nfarwtet or xtn2vmez).
       d) Typically, this requires 3 actions per child minimum (one prepare, one place on tray, one deliver).
    3. However, if sandwich preparation involves both actions kdzjcqwj and h0xkysus or tray swaps are needed,
       add a small overhead of 1-2 actions per child to account for complexity.
    4. We approximate uniform cost per child as 4 actions, providing a heuristic bias towards states with fewer remaining children to serve.
    5. The heuristic value is zero when all children have been served (all goals met), guaranteeing zero heuristic in goal states.
    6. The heuristic is simple and efficient, working directly on sets of facts; it's not guaranteed admissible but guides greedy best-first search effectively by focusing on remaining children.

    This heuristic balances computational efficiency, domain knowledge, and fidelity to action sequences observed in the domain.
    """

    def __init__(self, task):
        """Initialize by extracting goal children, static child types, and constants used in the domain."""
        self.goals = task.goals
        self.static = task.static

        # Extract all children mentioned in goal (ivo59apj ?child)
        self.goal_children = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts and parts[0] == "ivo59apj" and len(parts) == 2:
                self.goal_children.add(parts[1])

        # Extract all children of types n2ij1fah and dr25ifx3 (two child types)
        self.n2ij1fah_children = set()
        self.dr25ifx3_children = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and len(parts) == 2:
                pred, obj = parts
                if pred == "n2ij1fah":
                    self.n2ij1fah_children.add(obj)
                elif pred == "dr25ifx3":
                    self.dr25ifx3_children.add(obj)

        # Extract the unique constant of type zwhbr5cd: bupp7mhz
        # Normally given as constant in domain: (bupp7mhz - zwhbr5cd)
        # We'll treat that as a fixed constant used in matching tray predicates.
        self.unique_zwhbr5cd_const = "bupp7mhz"

        # Children to tables associations (eyker22y child table)
        self.child_to_table = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "eyker22y" and len(parts) == 3:
                child, table = parts[1], parts[2]
                self.child_to_table[child] = table

        # Extract all bread pieces and content objects from static for reference if needed (not strictly used here).
        # This could improve heuristic if desired.
        self.bread_objects = set()
        self.content_objects = set()
        for fact in task.facts:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "e6jz04yd":  # bread predicate
                    self.bread_objects.add(parts[1])
                elif parts[0] == "odd6wyd1":  # content predicate
                    self.content_objects.add(parts[1])

    def __call__(self, node):
        """Estimate number of remaining actions to reach goal from given state."""

        state = node.state

        # If all goal children are already served, heuristic = 0.
        served_children = {child for child in self.goal_children if f"(ivo59apj {child})" in state}
        remaining_children = self.goal_children - served_children
        if not remaining_children:
            return 0

        # Approximate 4 actions per remaining child as minimal heuristic cost:
        #  - one action to prepare sandwich (kdzjcqwj or h0xkysus)
        #  - one action to place sandwich on tray (jhzk46zy)
        #  - one action to deliver sandwich (nfarwtet or xtn2vmez)
        #  - an additional overhead to cover possible tray swaps (whgyhnl0) or nuances
        basic_actions_per_child = 4

        return basic_actions_per_child * len(remaining_children)
