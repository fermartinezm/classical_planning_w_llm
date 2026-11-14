from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the tokenized parts of a PDDL fact string, stripping surrounding parentheses.
    E.g. "(pred a b)" -> ["pred", "a", "b"].
    """
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic builds a simple relaxed planning graph (ignoring delete effects)
    starting from the current state and records the first layer in which each fact
    becomes available. It then estimates the required number of actions to reach
    the goal by summing the layer indices at which each goal fact first appears
    in the relaxed graph. The heuristic is 0 iff the state already satisfies all goals.

    Assumptions
    -----------
    - The planning task is STRIPS-like (operators have explicit preconditions,
      add effects and delete effects) as represented by the planner's Operator class.
    - Goals are conjunctive (a set of facts that must all hold).
    - Ignoring delete effects gives a fast, optimistic approximation of reachability.
    - Summing first-layer indices of individual goal facts produces a useful,
      admissible-or-not (not required) aggregate estimate for greedy best-first search.

    Heuristic Initialization
    ------------------------
    - Store references to the task's goals, static facts and operators.
    - Static facts (facts true in every state) are treated as initially available.
    - No further offline preprocessing is required for this relatively small domain.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the node's state already satisfies all goal facts, return 0.
    2. Initialize the relaxed-facts set with:
         - all facts currently true in the state, and
         - all static facts (facts true in every state).
       Record that these facts are available at layer 0.
    3. Repeatedly grow the relaxed fact set layer-by-layer:
         - Find all operators whose preconditions are a subset of the currently
           available relaxed facts.
         - For each such operator, add its add-effects to the relaxed facts.
         - For any newly added fact, record the layer index in which it first
           became available (current layer + 1).
       Stop when:
         - all goal facts are present (success), or
         - no new facts can be added in an iteration (failure to reach some goals).
    4. If all goals are reached in the relaxed graph:
         - Compute the heuristic value as the sum of the recorded first-layer indices
           for each goal fact. This reflects how many "layers" (i.e. steps of
           relaxed propagation) are needed to make each goal true.
    5. If some goals are unreachable in the relaxed graph, return a conservative
       finite estimate that grows with the number of missing goals and the
       number of layers explored. This avoids returning infinity while still
       steering search away from hopeless states.
    """

    def __init__(self, task):
        """Extract goal, static facts and operators from the task."""
        # Goals are a set of goal fact strings, e.g. "(gaijkwef a b)".
        self.goals = set(task.goals)
        # Static facts are facts that are true in every state (task.static).
        self.static_facts = set(task.static)
        # Operators are instances of the planner's Operator class.
        self.operators = list(task.operators)

    def __call__(self, node):
        """Compute the heuristic value for the given search node."""
        state = set(node.state)

        # 1) If goals already reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # 2) Initialize relaxed planning graph.
        relaxed_facts = set(state) | set(self.static_facts)
        first_layer = {fact: 0 for fact in relaxed_facts}  # layer index when fact first became true

        # Keep track of which operators we've already fired in the relaxed graph
        # to avoid redundant processing; firing an operator more than once never
        # produces new facts in the delete-free relaxation once its adds are included.
        fired_ops = set()

        layer = 0
        max_layers = len(self.operators) + 5  # safe cap to prevent pathological long loops
        # Iterate expanding layers until goals appear or no progress.
        while layer < max_layers:
            # If all goals are available, compute heuristic value as sum of first layers.
            if self.goals <= relaxed_facts:
                # Sum the layer indices when each goal first became available.
                # This gives a simple aggregated estimate of effort to achieve all goals.
                return sum(first_layer[goal] for goal in self.goals)

            newly_added = set()
            # Try to fire any operator whose preconditions are subset of relaxed_facts
            for op in self.operators:
                if op in fired_ops:
                    continue
                if op.preconditions <= relaxed_facts:
                    # This operator is applicable in the relaxed sense; add its add_effects.
                    for fact in op.add_effects:
                        if fact not in relaxed_facts and fact not in newly_added:
                            newly_added.add(fact)
                            # record the layer index when this fact first appears (layer+1)
                            first_layer[fact] = layer + 1
                    # mark operator fired; even if it added nothing new now, firing later won't help
                    fired_ops.add(op)

            if not newly_added:
                # No progress in this layer: some goals are unreachable in the relaxed graph.
                missing_goals = len([g for g in self.goals if g not in relaxed_facts])
                # Return a conservative finite heuristic:
                # number of missing goals multiplied by (layer+1) plus a penalty.
                # This gives larger values for more missing goals or deeper failure.
                return missing_goals * (layer + 1) + len(self.operators)

            # Update relaxed_facts and continue to next layer.
            relaxed_facts |= newly_added
            layer += 1

        # If we reached the max_layers without satisfying goals, return a large finite estimate.
        missing_goals = len([g for g in self.goals if g not in relaxed_facts])
        return missing_goals * (max_layers + 1) + len(self.operators)