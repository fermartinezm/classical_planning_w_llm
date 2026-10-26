# Assume 'heuristics.heuristic_base' module exists and contains the Heuristic base class.
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Define the miconicHeuristic class, inheriting from Heuristic
# class miconicHeuristic(Heuristic): # Uncomment this line in the actual environment
class miconicHeuristic: # Use this line for standalone code presentation
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the required number of actions by summing a fixed cost
    for each unserved passenger based on their current state (at origin or boarded).
    Passengers at their origin are estimated to need 4 actions (move to origin, board,
    move to destination, depart). Passengers who are boarded are estimated to need
    2 actions (move to destination, depart). This heuristic is non-admissible but
    designed to guide a greedy best-first search effectively by prioritizing states
    where passengers are closer to being served.

    # Assumptions
    - The domain follows the standard Miconic structure with 'origin', 'destin',
      'boarded', 'served', and 'lift-at' predicates, and 'board', 'depart', 'up',
      'down' actions.
    - The cost of each primitive action (move, board, depart) is 1.
    - The heuristic simplifies the problem by assuming passenger journeys are somewhat
      independent, ignoring potential optimizations from shared lift trips or optimal
      floor traversal sequences.

    # Heuristic Initialization
    The heuristic extracts the set of all passengers that need to be served by inspecting
    the goal conditions provided in the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value `h` to 0.
    2. Retrieve the set of all passengers who are required to be in the 'served' state
       in the goal. This set is determined during the heuristic's initialization by
       examining the task's goal facts.
    3. For each passenger `p` in this set:
       a. Check if the fact `(served p)` is present in the current state.
          If it is, this passenger has reached their goal, contributing 0 to the heuristic.
       b. If `(served p)` is not present, check if the fact `(boarded p)` is present.
          If `(boarded p)` is present, the passenger is currently inside the lift but
          not yet at their destination. They still need to be transported to their
          destination floor and depart. This is estimated to require 2 actions
          (specifically, the lift moving to the destination floor and the 'depart' action). Add 2 to `h`.
       c. If neither `(served p)` nor `(boarded p)` is present, the passenger must be
          at their origin floor. They need to be picked up at their origin, transported
          to their destination, and depart. This is estimated to require 4 actions
          (specifically, the lift moving to the origin floor, the 'board' action,
          the lift moving to the destination floor, and the 'depart' action). Add 4 to `h`.
    4. The total heuristic value `h` is the sum of these estimated costs for all
       passengers who are not yet served.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by identifying all passengers that need to be served.

        Args:
            task: The planning task object containing goals and static facts.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals

        # Extract all passengers mentioned in the goals (they all need to be served).
        # Assuming goal is a conjunction of (served ?p) facts.
        self.passengers_to_serve = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == 'served':
                self.passengers_to_serve.add(parts[1])

        # Static facts are available in task.static but not used in this simple heuristic.
        # self.static = task.static


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer estimate of the remaining actions to reach the goal.
        """
        state = node.state
        h = 0

        # Iterate through all passengers that need to be served according to the goal
        for passenger in self.passengers_to_serve:
            served_fact = f'(served {passenger})'
            boarded_fact = f'(boarded {passenger})'

            # Check if the passenger is served
            if served_fact in state:
                # Passenger is already served, contributes 0 to heuristic
                continue
            # Check if the passenger is boarded (and not served, due to the previous check)
            elif boarded_fact in state:
                # Passenger is boarded but not served
                # Estimated remaining actions: move to destination + depart = 2
                h += 2
            else:
                # Passenger is at origin (not served, not boarded)
                # Estimated remaining actions: move to origin + board + move to destination + depart = 4
                h += 4

        return h
