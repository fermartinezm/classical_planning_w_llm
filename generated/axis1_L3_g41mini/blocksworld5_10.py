from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (transport) domain.

    Summary
    -------
    This heuristic estimates the number of actions still needed to reach the goal by focusing on the 'road',
    'capacity-predecessor', 'capacity', 'in', and 'at' facts that represent the state of locatables.
    The domain models a transport-like task where objects must be driven or moved via sequences of "drive",
    "pick-up", "drop", and "drive-2" actions, manipulating facts about roads, capacities, positions, and predecessors.

    Assumptions
    -----------
    - The goal specifies certain 'road' facts, certain 'at' facts, and presence of 'capacity-predecessor' on some locatables.
    - Success relies on transitioning the correct locatables into 'road' status and moving them to the desired locations.
    - The sequence of actions relates to removing and adding 'road', 'capacity-predecessor', and 'capacity' facts
      while placing locatables at appropriate positions.
    - Each locatable that must become (road <loc>) and be at a proper <loc> needs to go through a chain of actions (pick-up, drive, drop, drive-2).
    - The predicate 'capacity' is a global state indicating if capacity is available for transitioning.

    Heuristic Initialization
    ------------------------
    - Store the sets of locatables required to have 'road' in the goal.
    - Store the (locatable, location) pairs required by the goal (for (at locatable location)).
    - Store locatables that must have 'capacity-predecessor' in the goal.
    - The heuristic relies only on these goal facts and current state facts; static facts are unused.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify all locatables that must eventually have the 'road' predicate true in the goal.
    2. Identify all (locatable, location) goal requirements from the 'at' predicates.
    3. Identify all locatables that must have 'capacity-predecessor' in the goal.
    4. For every locatable that must be 'road' at the end:
       - If it currently does not have 'road', estimate cost to get it there.
         The mechanism requires:
           a) 'pick-up' to produce the 'road' and 'capacity', 'capacity-predecessor' predicates from the locatable currently 'in'.
           b) 'drive' or 'drive-2' actions to move the locatable.
           c) 'drop' to set 'road' and update 'at' facts.
       - Actions can be chained, but we approximate costs by counting needed manipulations per locatable.
    5. For locatables needing to be at a specific location in the goal:
       - Check if the current 'at' fact matches the goal; if not, one or more moves will be needed.
       - Each move likely requires one or more 'drive-2' and 'drop' steps.
    6. For locatables needing 'capacity-predecessor' in the goal but currently missing it, additional actions needed.
    7. Estimate costs:
       - Each missing 'road' fact for a locatable: roughly 2 actions (pick-up + drive or drive-2).
       - Each locatable not at its goal location: roughly 1 to 2 actions (drive-2 + drop).
       - Each missing 'capacity-predecessor' fact: 1 action (pick-up or drop relevant).
       - Adjust estimate based on whether locatable is currently 'in' state or 'road' state.
    8. Sum these approximations for a total heuristic value.
    9. The heuristic is zero if all goal predicates hold.
    10. This heuristic is not admissible but aims to guide search efficiently.

    """

    def __init__(self, task):
        """Initialize the heuristic by extracting relevant goal conditions and static info."""
        self.goals = task.goals
        self.static = task.static  # not used here, but stored for completeness

        # Extract goal locatables that must have (road <loc>)
        self.goal_road_locs = set()
        # Extract locatable-location pairs from (at <locatable> <location>) goals
        self.goal_at_pairs = set()
        # Extract locatables that must have (capacity-predecessor <locatable>)
        self.goal_capacity_pred = set()

        for fact in self.goals:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "road" and len(parts) == 2:
                self.goal_road_locs.add(parts[1])
            elif pred == "at" and len(parts) == 3:
                self.goal_at_pairs.add((parts[1], parts[2]))
            elif pred == "capacity-predecessor" and len(parts) == 2:
                self.goal_capacity_pred.add(parts[1])

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of actions to achieve the goal from the given state.

        Parameters
        ----------
        node : search node object containing the current state as `node.state`

        Returns
        -------
        int
            Estimated action cost to the goal (heuristic value)
        """
        state = node.state

        # Helper sets from current state
        current_road = set()
        current_in = set()
        current_capacity_pred = set()
        current_at = dict()  # locatable -> location

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "road" and len(parts) == 2:
                current_road.add(parts[1])
            elif pred == "in" and len(parts) == 2:
                current_in.add(parts[1])
            elif pred == "capacity-predecessor" and len(parts) == 2:
                current_capacity_pred.add(parts[1])
            elif pred == "at" and len(parts) == 3:
                current_at[parts[1]] = parts[2]

        total_cost = 0

        # Estimate cost to fix locatables missing 'road' in the goal but not present now:
        # To get (road locatable), the locatable must be picked up from 'in' state (cost ~1),
        # then probably driven or dropped (roughly 1 more action).
        for loc in self.goal_road_locs:
            if loc not in current_road:
                # If locatable is currently 'in', one pick-up action needed at least
                if loc in current_in:
                    total_cost += 2  # pick-up + drive (approximation)
                else:
                    # If it's not 'in' or 'road', harder to say; count cost 3 to cover worst
                    total_cost += 3

        # Estimate cost for locatables that must be at certain locations in the goal:
        # If locatable not at goal location => roughly 1-2 actions to move/drop it properly.
        for locatable, goal_loc in self.goal_at_pairs:
            current_loc = current_at.get(locatable)
            if current_loc != goal_loc:
                # We assume 2 actions required: drive-2 and drop, or similar.
                total_cost += 2

        # Estimate cost for locatables that must have 'capacity-predecessor' but miss it:
        # This usually means pick-up or drop actions are needed.
        for loc in self.goal_capacity_pred:
            if loc not in current_capacity_pred:
                # Approximate with 1 action needed to restore
                total_cost += 1

        # Capacity predicate is global and required for some actions but we don't
        # explicitly track it here; it is implicitly managed when picking up or dropping.

        # If goals are satisfied exactly, heuristic should be zero
        if self.goals <= state:
            return 0

        return total_cost
