from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract predicate and arguments from a PDDL fact string.
    Example: '(s0899d5u b1 b2)' -> ['s0899d5u','b1','b2']"""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (z1spq3ra) domain.

    This heuristic produces a fast, informed estimate of the number of actions
    required to reach the goal from a given state. It is not guaranteed to be
    admissible, but aims to capture the main action patterns of the domain:
      - s0899d5u X Y facts (binary relation) are only created by ig1fewq3.
      - ig1fewq3 both creates a s0899d5u fact and typically (in the same action)
        establishes yfm3frbn on the subject (the first argument).
      - jjtnwk3y on an object is typically produced by cb5rcmgi (consuming ug241uf7
        and removing yfm3frbn and a4ha3ujl on that object) and consumed by
        ig1fewq3; dezigdo8 restores yfm3frbn and a4ha3ujl and ug241uf7 but
        requires jjtnwk3y as a precondition.
      - ug241uf7 is a global toggle used by many actions.

    Assumptions
    -----------
    - The planner represents facts as strings "(predicate arg1 arg2 ...)".
    - Goals are conjunctive facts in task.goals (set of strings).
    - The most common way to create a missing s0899d5u(e,q) is a short sequence
      that includes producing jjtnwk3y(e) (via cb5rcmgi) and then calling ig1fewq3(e,q).
      We account for those interactions conservatively by assigning a small fixed
      cost for s-creation that also acknowledges that ig1fewq3 creates yfm3frbn(e).
    - The heuristic trades accuracy for speed and uses simple domain knowledge
      (which facts are created together) to avoid large search expansions.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and organize the goal facts (by predicate)
    into fast-to-query sets:
      - self.goal_y : set of objects that must have (yfm3frbn OBJ)
      - self.goal_a : set of objects that must have (a4ha3ujl OBJ)
      - self.goal_j : set of objects that must have (jjtnwk3y OBJ)
      - self.goal_s : set of pairs (SUBJ,OBJ) that must have (s0899d5u SUBJ OBJ)
      - self.goal_ug: boolean (True if (ug241uf7) is a goal)

    Static facts (task.static) are recorded (even if empty), to satisfy the
    requirement of extracting static information; this domain has no static facts
    in the provided examples but the field is saved for completeness.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. Quick goal-check: if all goals are already present in the state, return 0.

    2. Parse the current state into predicate-indexed sets:
       - which objects presently have yfm3frbn, a4ha3ujl, jjtnwk3y
       - which s0899d5u pairs are present
       - whether ug241uf7 is present globally

    3. Identify which goal facts are missing (goals \ current_state).
       We treat missing goals of different predicates with different costs,
       reflecting the typical number of actions required to achieve them:
         - missing s0899d5u(subj,obj): cost 2 (heuristic estimate: cb5rcmgi(subj) + ig1fewq3(subj,obj)).
             Note: ig1fewq3 concurrently introduces yfm3frbn(subj). To avoid double
             counting, if yfm3frbn(subj) is also a missing goal we consider it
             satisfied by the same s-creation action (and do not add an extra cost).
         - missing yfm3frbn(obj): cost 1 (can be produced by dezigdo8 or ig1fewq3/i57yke4t
             in typical short sequences).
         - missing a4ha3ujl(obj): cost 1 (dezigdo8 produces a4ha3ujl, counted optimistically).
         - missing jjtnwk3y(obj): cost 1 (cb5rcmgi is the typical way to produce jjtnwk3y).
         - missing ug241uf7: cost 1 (many actions toggle ug241uf7; treat as a small cost).

    4. Greedy combined-effect accounting:
       - Process missing s0899d5u goals first and mark any yfm3frbn(subj) missing
         goals as satisfied by the s-creation action (since ig1fewq3 adds that fact).
       - Then add costs for remaining missing y/a/j/ug goals.

    5. Return the summed integer estimate.

    Notes on Heuristic Behavior
    ---------------------------
    - The heuristic is intentionally simple and fast to compute. It captures
      main coupling between s0899d5u and yfm3frbn facts (created together) which
      helps avoid overestimating when both are required and thus reduces search
      expansions in many typical instances.
    - The heuristic is not admissible: some counted actions may be achievable
      simultaneously in fewer operator applications in certain states.
    - The heuristic always returns a finite integer for any finite goal set.
    """

    def __init__(self, task):
        # Record goals and static facts
        self.goals = set(task.goals)
        self.static = set(task.static)

        # Initialize goal partitions
        self.goal_y = set()
        self.goal_a = set()
        self.goal_j = set()
        self.goal_s = set()
        self.goal_ug = False

        for g in self.goals:
            parts = get_parts(g)
            pred = parts[0]
            if pred == "yfm3frbn":
                self.goal_y.add(parts[1])
            elif pred == "a4ha3ujl":
                self.goal_a.add(parts[1])
            elif pred == "jjtnwk3y":
                self.goal_j.add(parts[1])
            elif pred == "s0899d5u":
                self.goal_s.add((parts[1], parts[2]))
            elif pred == "ug241uf7":
                self.goal_ug = True

    def __call__(self, node):
        state = node.state

        # If goal already satisfied, heuristic is 0
        if self.goals <= state:
            return 0

        # Parse current state into predicate-based structures
        y_set = set()
        a_set = set()
        j_set = set()
        s_set = set()
        ug_present = False

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "yfm3frbn":
                y_set.add(parts[1])
            elif pred == "a4ha3ujl":
                a_set.add(parts[1])
            elif pred == "jjtnwk3y":
                j_set.add(parts[1])
            elif pred == "s0899d5u":
                s_set.add((parts[1], parts[2]))
            elif pred == "ug241uf7":
                ug_present = True

        # Determine which goal facts are missing
        missing_s = set(self.goal_s - s_set)
        missing_y = set(self.goal_y - y_set)
        missing_a = set(self.goal_a - a_set)
        missing_j = set(self.goal_j - j_set)
        missing_ug = self.goal_ug and (not ug_present)

        cost = 0

        # Heuristic weights (domain-informed, small integer values)
        # Creating an s0899d5u typically requires creating jj on subject then calling ig1fewq3:
        S_COST = 2
        # Creating yfm3frbn or a4ha3ujl or jjtnwk3y or ug241uf7 counted cheaply:
        Y_COST = 1
        A_COST = 1
        J_COST = 1
        UG_COST = 1

        # Process s-goals first. ig1fewq3 both creates s and y(subject),
        # so if y(subject) is also a missing goal, we treat it as satisfied by the s-creation.
        for (subj, obj) in missing_s:
            cost += S_COST
            # If y(subject) was also a missing goal, mark it satisfied by this s-creation.
            if subj in missing_y:
                missing_y.remove(subj)

        # Now add remaining missing unary goals
        cost += len(missing_y) * Y_COST
        cost += len(missing_a) * A_COST
        cost += len(missing_j) * J_COST
        if missing_ug:
            cost += UG_COST

        # Return integer heuristic value
        return int(cost)