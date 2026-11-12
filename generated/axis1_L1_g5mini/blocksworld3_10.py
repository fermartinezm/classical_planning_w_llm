from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the parts of a PDDL fact string without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-specific heuristic for the secret_domain (z1spq3ra). The domain manipulates
    three unary predicates on objects (yfm3frbn, a4ha3ujl, jjtnwk3y), a global boolean
    predicate (ug241uf7) and a binary relation (s0899d5u). Actions create or remove
    s0899d5u links while moving the unary markers between objects and toggling the
    global flag. The heuristic estimates the number of actions required to satisfy
    the goal facts (which typically include unary markers and s0899d5u edges).

    Assumptions
    -----------
    - The planner represents facts as strings like "(pred obj ...)" and task.goals
      contains the goal facts as such strings.
    - The domain's important facts for goals are only:
        - (yfm3frbn <obj>)
        - (a4ha3ujl <obj>)
        - (s0899d5u <src> <tgt>)
      (Other facts may exist but are not typically required directly in the goals.)
    - Actions that create facts often also create/consume other related facts (so a
      single action can satisfy multiple goal atoms). The heuristic tries to exploit
      simple opportunities where one action satisfies two unary goals at once.
    - The heuristic need not be admissible; it may overestimate somewhat to improve
      guidance, but it always returns a finite integer for solvable states and 0
      exactly when all goals are satisfied.

    Heuristic Initialization
    ------------------------
    The constructor extracts and stores:
    - self.goals : the frozenset of goal fact strings
    - self.goal_yfm : set of object names that must satisfy (yfm3frbn <obj>)
    - self.goal_a4h : set of object names that must satisfy (a4ha3ujl <obj>)
    - self.goal_s   : set of (src, tgt) tuples for required (s0899d5u src tgt)
    - self.static   : task.static (kept for completeness; this domain does not use static)
    These are computed once so the heuristic can quickly compare the current state to goals.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    The heuristic works as follows for a given node.state:

    1. Quick goal check:
       - If all goal facts are already present in the state, return 0.

    2. Parse current facts (sets):
       - cur_yfm   : objects with (yfm3frbn <obj>) in the state
       - cur_a4h   : objects with (a4ha3ujl <obj>) in the state
       - cur_jj    : objects with (jjtnwk3y <obj>) in the state
       - cur_s     : set of (src, tgt) tuples for s0899d5u facts present
       - The algorithm only uses these unary/binary facts to estimate minimal actions.

    3. Compute missing goal atoms:
       - missing_yfm = goal_yfm - cur_yfm
       - missing_a4h = goal_a4h - cur_a4h
       - missing_s   = goal_s - cur_s

    4. Exploit multi-effect actions:
       - If for the same object both (yfm3frbn obj) and (a4ha3ujl obj) are goals and
         both are currently missing, then there is an action (dezigdo8) that can
         add both facts simultaneously if jjtnwk3y obj holds. Thus:
           - If jjtnwk3y already present => cost 1 to satisfy both
           - Otherwise conservatively cost 2 (one action to create/obtain jjtnwk3y,
             and one to apply dezigdo8) and assume both facts will be produced.
       - When we account for these planned creations they are added to the
         "available" current facts so subsequent estimates (e.g., for s-edge creation)
         can use them.

    5. Remaining single unary goals:
       - A single missing yfm3frbn or a4ha3ujl is estimated as cost 1 to create (a
         single action can make it in common situations). After counting such a
         creation we also mark the fact as available for later steps.

    6. Estimating s0899d5u creation cost:
       - To create s0899d5u (src tgt) an action such as ig1fewq3 can be used which
         requires jjtnwk3y on src and yfm3frbn on tgt. We therefore estimate:
           cost_for_s = 1 + (1 if jjtnwk3y(src) not yet available else 0)
                           + (1 if yfm3frbn(tgt) not yet available else 0)
         - This yields cost 1 when both preconditions are already available,
           cost 2 if one precondition is missing, and cost up to 3 if both are missing.
       - We update the available facts after accounting for the s-creation estimate,
         reflecting that creating the s-edge will often also restore or move yfm3frbn
         markers (the heuristic already counted/assumed the necessary marker creations
         as part of the cost when needed).

    7. Sum all the estimated costs and return the total as an integer.

    Rationale
    ---------
    - The heuristic is deliberately simple and local: it counts missing goal atoms
      but reduces cost when a single action can achieve multiple goal atoms (observed
      in the domain operators). This is intended to provide informative, cheaply
      computed guidance for greedy best-first search while remaining fast.
    - The heuristic avoids expensive global reasoning (search in the relaxed space).
      It attempts to reflect necessary preconditions in a light-weight manner so
      that states closer to providing required preconditions (like having jjtnwk3y
      on the correct object or yfm3frbn on the target) are given lower cost.
    """

    def __init__(self, task):
        """Extract static information and goals from the task."""
        self.goals = task.goals
        self.static = task.static

        # Parse goal facts into structured sets.
        self.goal_yfm = set()
        self.goal_a4h = set()
        self.goal_s = set()

        for fact in self.goals:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "yfm3frbn" and len(parts) >= 2:
                self.goal_yfm.add(parts[1])
            elif pred == "a4ha3ujl" and len(parts) >= 2:
                self.goal_a4h.add(parts[1])
            elif pred == "s0899d5u" and len(parts) >= 3:
                self.goal_s.add((parts[1], parts[2]))

    def __call__(self, node):
        """Estimate the number of actions required to reach a goal state from node.state."""
        state = node.state

        # Fast check: if all goals already satisfied, return 0.
        if self.goals <= state:
            return 0

        # Build current fact sets.
        cur_yfm = set()
        cur_a4h = set()
        cur_jj = set()
        cur_s = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "yfm3frbn" and len(parts) >= 2:
                cur_yfm.add(parts[1])
            elif pred == "a4ha3ujl" and len(parts) >= 2:
                cur_a4h.add(parts[1])
            elif pred == "jjtnwk3y" and len(parts) >= 2:
                cur_jj.add(parts[1])
            elif pred == "s0899d5u" and len(parts) >= 3:
                cur_s.add((parts[1], parts[2]))

        # Missing goal atoms.
        missing_yfm = set(self.goal_yfm - cur_yfm)
        missing_a4h = set(self.goal_a4h - cur_a4h)
        missing_s = set(self.goal_s - cur_s)

        cost = 0

        # Step 1: Handle objects that need both yfm3frbn and a4ha3ujl.
        # A single dezigdo8 (if jj present) can satisfy both; otherwise estimate 2.
        both_missing = missing_yfm & missing_a4h
        for obj in list(both_missing):
            if obj in cur_jj:
                # One action (dezigdo8) can add both yfm3frbn and a4ha3ujl.
                cost += 1
                # Mark them as planned/available.
                cur_yfm.add(obj)
                cur_a4h.add(obj)
            else:
                # Conservatively assume two actions: one to obtain jjtnwk3y (or another
                # sequence) and one to apply dezigdo8 that adds both facts.
                cost += 2
                cur_yfm.add(obj)
                cur_a4h.add(obj)
            missing_yfm.discard(obj)
            missing_a4h.discard(obj)

        # Step 2: Handle remaining single missing unary goals (each approx cost 1).
        for obj in list(missing_yfm):
            cost += 1
            cur_yfm.add(obj)
            missing_yfm.discard(obj)

        for obj in list(missing_a4h):
            cost += 1
            cur_a4h.add(obj)
            missing_a4h.discard(obj)

        # Step 3: Handle missing s0899d5u facts.
        # To create s(src, tgt) we typically need jjtnwk3y(src) and yfm3frbn(tgt),
        # and then one action (ig1fewq3 or similar) creates the s-edge. We estimate:
        # cost_for_s = 1 + (1 if jj missing) + (1 if yfm missing)
        for (src, tgt) in missing_s:
            pre_src_ok = src in cur_jj
            pre_tgt_ok = tgt in cur_yfm
            # One action to create s plus actions to create missing preconditions.
            s_cost = 1 + (0 if pre_src_ok else 1) + (0 if pre_tgt_ok else 1)
            cost += s_cost
            # Assume after doing this the s edge exists and the source/target markers
            # are updated in a way that doesn't prevent other planned actions:
            cur_s.add((src, tgt))
            # Also reflect plausible side-effects: after creating s via ig1fewq3,
            # the source often receives yfm3frbn; we conservatively keep cur_yfm as-is
            # because earlier steps already added yfm when needed.

        # Return the aggregate estimated action cost.
        # Ensure finite integer is returned even if there are no goals (handled above).
        return int(cost)