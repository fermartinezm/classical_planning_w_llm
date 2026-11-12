from heuristics.heuristic_base import Heuristic


def parse_fact(fact):
    """Parse a PDDL fact string into predicate and list of objects."""
    parts = fact[1:-1].split()
    return parts[0], parts[1:]


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (domain z1spq3ra).

    Summary:
    This heuristic estimates the number of actions required to reach the goal 
    state from the current state by focusing on how the predicates (yfm3frbn), 
    (a4ha3ujl), (jjtnwk3y), and (s0899d5u) must align to reach the goal. The 
    main crux is that enabled transitions depend on the unique Boolean predicate 
    (ug241uf7), which toggles between true and false, and on the various facts 
    for each object. The heuristic accounts for the minimal number of transformations 
    needed to establish the required links between objects to fulfill the goal. 
    
    It approximates the cost by analyzing differences between the current and goal 
    facts related to these predicates, counting necessary conversions of object 
    states and the required (s0899d5u) links to satisfy the goal. 
    
    Assumptions:
    - The domain has objects whose states are defined by predicates yfm3frbn, a4ha3ujl, jjtnwk3y.
    - The predicate ug241uf7 acts as a global toggle flag needed for certain actions.
    - The (s0899d5u x y) predicate defines a relationship between two objects.
    - Only one (ug241uf7) fact exists in the state at a time.
    - The heuristic counts needed object state "mode" changes and links missing to reach goal.
    - Exact minimal number of actions is difficult, so heuristic is an approximation that 
      guides greedy best-first search efficiently without guarantee of admissibility.

    Heuristic Initialization:
    - Extract goal facts separating into sets:
      * goal_yfm3frbn (objects that must hold (yfm3frbn))
      * goal_a4ha3ujl (objects that must hold (a4ha3ujl))
      * goal_jjtnwk3y (objects that must hold (jjtnwk3y))
      * goal_s0899d5u (object pairs (obj1, obj2) that must hold (s0899d5u obj1 obj2))
    - Extract the set of all objects appearing in goals for the predicates above.

    Step-By-Step Thinking for Computing Heuristic:
    1. If current state satisfies all goal facts, heuristic is 0.

    2. Identify current sets of objects that hold predicates yfm3frbn, a4ha3ujl, and jjtnwk3y. 

    3. Identify current (s0899d5u) links.

    4. Count how many goal (yfm3frbn), (a4ha3ujl), (jjtnwk3y) facts are not matched in current state.
       Each unmatched predicate for an object implies at least one action to fix, since predicates 
       are toggled in actions.

    5. Count how many (s0899d5u) goal links are missing in the current state.

    6. The predicate (ug241uf7) toggles with every action involving state changes, so estimate 
       additional cost related to its toggle: generally, fixing states and links requires toggling it.

    7. Combine these differences into a heuristic estimate:
       heuristic := sum of unmatched predicate facts + number of missing links

    This heuristic reflects the number of predicate "mode" changes and link creations 
    that must occur to reach the goal, aiming to efficiently guide the search.

    """

    def __init__(self, task):
        self.goals = task.goals
        # Parse goals into predicate-object mappings for fast lookup
        self.goal_yfm3frbn = set()
        self.goal_a4ha3ujl = set()
        self.goal_jjtnwk3y = set()
        self.goal_s0899d5u = set()
        for goal_fact in self.goals:
            pred, objs = parse_fact(goal_fact)
            if pred == "yfm3frbn":
                self.goal_yfm3frbn.add(objs[0])
            elif pred == "a4ha3ujl":
                self.goal_a4ha3ujl.add(objs[0])
            elif pred == "jjtnwk3y":
                self.goal_jjtnwk3y.add(objs[0])
            elif pred == "s0899d5u":
                self.goal_s0899d5u.add((objs[0], objs[1]))

    def __call__(self, node):
        state = node.state

        # Quick check if goal already satisfied (heuristic=0)
        if self.goals <= state:
            return 0

        # Current sets of objects for each predicate
        curr_yfm3frbn = set()
        curr_a4ha3ujl = set()
        curr_jjtnwk3y = set()
        curr_s0899d5u = set()
        ug241uf7_present = False

        for fact in state:
            pred, objs = parse_fact(fact)
            if pred == "yfm3frbn" and objs:
                curr_yfm3frbn.add(objs[0])
            elif pred == "a4ha3ujl" and objs:
                curr_a4ha3ujl.add(objs[0])
            elif pred == "jjtnwk3y" and objs:
                curr_jjtnwk3y.add(objs[0])
            elif pred == "s0899d5u" and len(objs) == 2:
                curr_s0899d5u.add((objs[0], objs[1]))
            elif pred == "ug241uf7":
                ug241uf7_present = True

        # Count missing predicate facts for objects in goal
        missing_yfm3frbn = len(self.goal_yfm3frbn - curr_yfm3frbn)
        missing_a4ha3ujl = len(self.goal_a4ha3ujl - curr_a4ha3ujl)
        missing_jjtnwk3y = len(self.goal_jjtnwk3y - curr_jjtnwk3y)

        # Count missing s0899d5u links
        missing_links = len(self.goal_s0899d5u - curr_s0899d5u)

        # Each mismatch in predicate facts typically requires one or more actions (with ug241uf7 toggling)
        # Since actions affect ug241uf7, if predicate facts are wrong, at least one toggle action will be needed.

        # Sum the differences as the heuristic value.
        heuristic_value = missing_yfm3frbn + missing_a4ha3ujl + missing_jjtnwk3y + missing_links

        # To reflect the cost of toggling (ug241uf7), add one if current ug241uf7 differs 
        # from required in goal or to handle actions needing it.
        # The goal does not specify ug241uf7, but its presence affects action applicability.
        # Here we conservatively do not add extra cost for ug241uf7 toggling, since estimated 
        # by predicate mismatches count.

        # Ensure minimal heuristic is at least 1 if not goal state (to avoid zero heuristic except goal)
        if heuristic_value == 0:
            heuristic_value = 1

        return heuristic_value
