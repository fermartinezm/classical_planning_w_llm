from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split the fact into tokens."""
    return fact[1:-1].split()


def fact_str(pred, *args):
    """Reconstruct a fact string from predicate and args (matching planner format)."""
    return f"({pred}{(' ' + ' '.join(args)) if args else ''})"


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent, fast-to-compute heuristic for the 'transport' domain (called
    secret_domain in the task). The heuristic estimates the number of actions
    required to reach the goal by inspecting which goal facts are currently
    missing and assigning a small, domain-informed cost to each missing fact.
    The estimate is not guaranteed to be admissible, but it is efficient and
    attempts to reflect operator dependencies (for example, that an 'at' fact
    typically needs an 'in' fact and a 'road' fact as preconditions of the
    operator that produces it).

    Assumptions
    -----------
    - The planner represents facts as strings of the form "(predicate arg1 arg2 ...)".
    - The main goal-relevant predicates in this domain are:
      - at <obj> <loc>
      - road <obj>
      - capacity-predecessor <obj>
      - capacity
      - in <obj>
    - An 'at' goal typically requires a DROP action which needs the carried status
      ("in <obj>") and the destination being on the road ("road <dest>"). We thus
      charge extra cost for missing supporting facts.
    - A 'road' or 'capacity-predecessor' fact is usually created by an action that
      also depends on the object being "in" or on other supporting facts; the
      heuristic charges a small extra cost when those supporting facts are not
      currently available.
    - Static facts (task.static) are extracted and available though for this
      domain they are usually empty. The constructor still retains the static
      facts for future extension.

    Heuristic Initialization
    ------------------------
    - self.goals: the set of goal facts (strings) as provided by the Task.
    - self.static_facts: the set of static facts provided by the Task (kept for any
      potential use, even though this domain typically contains no static facts).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node/state:
    1. If the state's facts already contain all goal facts, return 0.
    2. Parse the current state's facts into a mapping from predicate -> set(args-tuples)
       to allow quick membership queries about supporting facts (e.g., is "(in obj)"
       true? is "(road obj)" true?).
    3. For each goal fact that is not present in the current state, add a small,
       goal-specific estimated cost:
       - at obj dest:
         Base cost = 1 (drop action). If "(in obj)" is missing, add +1 (must first
         produce the 'in' status). If "(road dest)" is missing, add +1 (destination
         must be on road). So missing an 'at' fact typically costs between 1 and 3.
       - road obj:
         If "(road obj)" missing but "(in obj)" present, cost = 1 (pick-up can
         create road for that object). Otherwise cost = 2 (needs making the object
         'in' and then an action to create the road).
       - capacity-predecessor obj:
         If "(capacity-predecessor obj)" missing but "(in obj)" present, cost = 1
         (a pick-up can create it), otherwise cost = 2 (need to make it 'in' first).
       - capacity:
         If missing and some "(in ...)" exists in the state, cost = 1 (a pick-up
         or drop will create capacity); otherwise cost = 2.
       - in obj:
         If missing, we conservatively estimate cost = 1 when "(capacity)" present
         (a drive or drive-2 can make an object 'in' if other conditions hold),
         otherwise cost = 2. This keeps the estimate finite and small while
         acknowledging the typical need for capacity.
       - Any other predicate (default):
         Assign a small unit cost of 1 for missing facts.
    4. Sum the costs for all missing goal facts and return that sum as the heuristic.
    Notes:
    - The heuristic is intentionally simple: it trades exactness for speed and
      robustness. It captures key dependencies (for example, that an 'at' often
      requires 'in' and a 'road' fact) so it tends to prefer states that have
      necessary supporting facts already established.
    - Because multiple missing goals may share supporting facts, the heuristic can
      overcount; this is acceptable for greedy best-first search guidance and
      keeps computation fast.
    """

    def __init__(self, task):
        """Extract goals and static facts for quick access."""
        self.goals = set(task.goals) if task.goals is not None else set()
        # Keep static facts in case they are meaningful in instance variants.
        self.static_facts = set(task.static) if task.static is not None else set()

    def __call__(self, node):
        """Compute and return the heuristic estimate for the given search node."""
        state = node.state

        # If the goal is already reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # Build predicate -> set of args tuples mapping for quick membership checks.
        preds = {}
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            args = tuple(parts[1:])
            preds.setdefault(pred, set()).add(args)

        def has_fact(pred, *args):
            """Check whether the fact (pred args...) is present in the current state."""
            return tuple(args) in preds.get(pred, set())

        total_cost = 0

        for goal in self.goals:
            parts = get_parts(goal)
            pred = parts[0]
            args = parts[1:]

            # Skip facts already satisfied.
            if tuple(args) in preds.get(pred, set()):
                continue

            # Estimate costs per predicate using small domain-informed rules.
            if pred == "at":
                # A DROP produces (at obj dest). DROP requires (in obj) and (road dest).
                obj, dest = args
                cost = 1  # base cost for the drop action
                if not has_fact("in", obj):
                    cost += 1  # need to get obj 'in' first
                if not has_fact("road", dest):
                    cost += 1  # need destination to be on road
                total_cost += cost

            elif pred == "road":
                (obj,) = args
                # road can be produced by pick-up if the object is currently 'in'.
                if has_fact("road", obj):
                    # shouldn't happen because we checked presence above, but keep safe
                    continue
                if has_fact("in", obj):
                    total_cost += 1
                else:
                    total_cost += 2

            elif pred == "capacity-predecessor":
                (obj,) = args
                # Usually created by pick-up, which requires the object to be 'in'.
                if has_fact("capacity-predecessor", obj):
                    continue
                if has_fact("in", obj):
                    total_cost += 1
                else:
                    total_cost += 2

            elif pred == "capacity":
                if has_fact("capacity"):
                    continue
                # pick-up or drop create capacity; if any object is currently 'in'
                # then likely one of those actions can be applied soon.
                if preds.get("in"):
                    total_cost += 1
                else:
                    total_cost += 2

            elif pred == "in":
                (obj,) = args
                # 'in' is produced by drive or drive-2 which typically require capacity.
                if has_fact("capacity"):
                    total_cost += 1
                else:
                    total_cost += 2

            else:
                # Generic fallback: small unit cost for missing goal facts.
                total_cost += 1

        # Ensure a finite integer is returned.
        return int(total_cost)