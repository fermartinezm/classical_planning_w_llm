from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain ('transport') domain.

    Summary
    -------
    Estimates the number of actions required to reach the goal state from the current state.
    The domain models a transport process with facts about roads, capacity, in/out status,
    and 'at' relationships capturing locations of entities. Key predicates like:
    - (road ?v)
    - (capacity-predecessor ?v)
    - (capacity)
    - (in ?v)
    - (at ?v1 ?v2)
    describe the current transport state, and goal states impose certain conditions on these.

    The heuristic is designed to guide greedy best-first search by estimating how many
    actions are needed to convert the current state into one that satisfies the goal facts.

    Assumptions
    -----------
    - The goal facts must all be satisfied for a goal state.
    - The predicates and facts revolve around transitions between objects via "road"
      connections and capacity predicates that are toggled by pick-up, drive, drop,
      and drive-2 actions.
    - Each "road" fact corresponds to some transportable entity being on the road.
    - The "capacity" and "capacity-predecessor" predicates indirectly express the capacity 
      availability in the system.
    - The transport is essentially about moving objects along a chain of locations via "at"
      and "road" predicates.
    - The cost of changing each relevant fact towards the goal is roughly one action.
    - Actions tend to fix one predicate at a time; thus, the heuristic aggregates discrepancies
      in key predicates to estimate the number of needed actions.

    Heuristic Initialization
    ------------------------
    - The heuristic stores the goal facts to check which predicates should hold.
    - No static facts are used from the task since the domain does not provide static info.
    - The heuristic will analyze the state by scanning predicates:
      * road facts
      * capacity-predecessor facts
      * capacity predicate (single boolean predicate)
      * in predicate (objects currently "in")
      * at facts indicating object locations

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state satisfies all goal facts, return 0.
    2. Compute the number of goal facts not satisfied in the current state.
    3. Differentiate predicate types to avoid overcounting:
       - Count how many goal road facts are missing.
       - Count how many capacity-predecessor facts are missing.
       - Check if capacity predicate is missing.
       - Check if 'in' predicate fact(s) in goal are missing.
       - Count how many goal 'at' facts are missing.
    4. Each missing predicate is assumed to require (at least) one action to fix.
       The heuristic returns the sum of these missing predicates.
    5. This summation gives a finite, informative, and efficiently computable estimate
       that usually reflects the minimum number of actions to reach the goal in practice,
       though it is not guaranteed admissible (as allowed).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing the goal facts.
        Static facts are not used since none are defined for this domain.
        """
        self.goals = task.goals
        self.static = task.static  # Not used but kept for interface consistency.

        # Partition goal facts by predicate for efficient checking in __call__.
        self.goal_road = set()
        self.goal_capacity_predecessor = set()
        self.goal_capacity = False
        self.goal_in = set()
        self.goal_at = set()

        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "road":
                self.goal_road.add(fact)
            elif pred == "capacity-predecessor":
                self.goal_capacity_predecessor.add(fact)
            elif pred == "capacity":
                self.goal_capacity = True
            elif pred == "in":
                self.goal_in.add(fact)
            elif pred == "at":
                self.goal_at.add(fact)

    def __call__(self, node):
        """
        Compute an estimate of the minimum number of actions required from the current state.
        """
        state = node.state

        # If state already satisfies goals, heuristic is zero.
        if self.goals <= state:
            return 0

        cost = 0

        # Missing 'road' goals
        missing_road = len(self.goal_road - state)
        cost += missing_road

        # Missing 'capacity-predecessor' goals
        missing_cap_pred = len(self.goal_capacity_predecessor - state)
        cost += missing_cap_pred

        # Missing 'capacity' predicate
        if self.goal_capacity and "(capacity)" not in state:
            cost += 1

        # Missing 'in' predicates
        missing_in = len(self.goal_in - state)
        cost += missing_in

        # Missing 'at' predicates
        missing_at = len(self.goal_at - state)
        cost += missing_at

        return cost if cost > 0 else 1  # At least 1 if not goal to avoid zero heuristic in non-goal states
