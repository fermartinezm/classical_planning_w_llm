from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(i9fq3vit sandw3)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve all goal facts of the form (ivo59apj ?child),
    which represent that a particular 'bciid6ad' object (child) has received a sandwich via a complicated preparation 
    and serving process. Actions involve preparing sandwiches from bread and content, delivering sandwiches on trays 
    to children, and satisfying goal conditions of children having been served.

    Assumptions
    -----------
    - The goal is fully characterized by a set of facts (ivo59apj ?child).
    - Serving a child requires that the corresponding fact (ivo59apj ?child) be true.
    - The heuristic counts unfinished goals (children not served) and estimates the minimal number of actions required
      to serve each child individually.
    - Interactions between serving different children are treated independently for heuristic simplicity.
    - The main bottleneck is preparing and delivering sandwiches for each child.
    - Sandwich preparation either uses action 'kdzjcqwj' or 'h0xkysus' depending on available preconditions.
    - Sandwich serving involves actions 'jhzk46zy' and then either 'nfarwtet' or 'xtn2vmez'.
    - Trays can be moved between tables or locations with 'whgyhnl0' if needed, but this cost is ignored for efficiency.

    Heuristic Initialization
    ------------------------
    From the static facts and goals, we extract:
    - The set of children to be served (from goal facts like (ivo59apj ?child))
    - Mapping of children ('bciid6ad') to their associated location ('zwhbr5cd') via (eyker22y ?child ?loc)
    - The single constant of type zwhbr5cd (bupp7mhz), representing a special location used in conditions
    - For efficiency we store:
      * Sets of children typed as (n2ij1fah ?child) and (dr25ifx3 ?child)
      * Locations of trays currently on bupp7mhz using facts (y0rxzzmc ?tray bupp7mhz)

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Determine which children still need to be served (goal fact absent in current state).
    2. For each unserved child:
       a) The sandwich to serve them must be prepared and delivered.
       b) Preparing a sandwich requires a sandwich object (?yam4s):
          - To prepare, a sandwich must become (i9fq3vit ?yam4s), which happens after applying either
            'kdzjcqwj' or 'h0xkysus'.
          - The sandwich starts from (zogo0zhl ?yam4s) and becomes (i9fq3vit ?yam4s) after preparation.
       c) Then the sandwich must be placed on a tray:
          - 'jhzk46zy' moves sandwich from (i9fq3vit) status to (mpvvhjpt) status tied to a tray on bupp7mhz.
       d) Finally, delivering to the child requires either:
          - If child has (n2ij1fah) typing, 'nfarwtet' moves delivered sandwich (removes mpvvhjpt) and adds (ivo59apj).
          - If child has (dr25ifx3), 'xtn2vmez' action does the same.
    3. Each child must have these phases completed for (ivo59apj ?child) to hold:
       - Prepare sandwich: 1 or 2 actions (kdzjcqwj or h0xkysus)
       - Place on tray: 1 action (jhzk46zy)
       - Deliver (finalize): 1 action (nfarwtet or xtn2vmez)
    4. Ignore the cost of tray rearrangement actions (whgyhnl0), assuming trays are available at bupp7mhz.
    5. Estimate per child minimal number of actions as:
       - If we assume worst case both preparation actions needed (likely just one),
         count max 2 prep actions.
       - But domain has two alternative preparation actions:
         - 'kdzjcqwj' requires more preconditions but more effects.
         - 'h0xkysus' simpler.
       For heuristic, assume 1 prep action per sandwich suffices (choose minimal).
    6. Sum up minimum estimated cost for all unserved children.
    7. Return 0 if goal achieved (no child left unserved).

    This heuristic is not guaranteed admissible but guides the search by estimating minimal required action sequences
    per child to reach (ivo59apj ?child).
    """

    def __init__(self, task):
        # Extract goal children to be served: from (ivo59apj ?child) goals
        self.goal_children = set()
        for g in task.goals:
            parts = get_parts(g)
            if parts[0] == "ivo59apj" and len(parts) == 2:
                self.goal_children.add(parts[1])

        static_facts = task.static

        # Identify child sets by type predicates (n2ij1fah and dr25ifx3)
        self.n2ij1fah_children = {
            get_parts(f)[1] for f in static_facts if match(f, "n2ij1fah", "*")
        }
        self.dr25ifx3_children = {
            get_parts(f)[1] for f in static_facts if match(f, "dr25ifx3", "*")
        }

        # Map children to zwhbr5cd location via (eyker22y ?child ?loc)
        self.child_location = {}
        for fact in static_facts:
            if match(fact, "eyker22y", "*", "*"):
                parts = get_parts(fact)
                self.child_location[parts[1]] = parts[2]

        # Extract the constant of type zwhbr5cd: bupp7mhz (only one in domain)
        # We can hardcode "bupp7mhz" as it is known constant, but safer to confirm
        self.special_location = "bupp7mhz"

    def __call__(self, node):
        state = node.state

        # Quick check: if goal reached, heuristic is 0
        # Goal: all (ivo59apj ?child) goal children must be present
        if all(f"(ivo59apj {child})" in state for child in self.goal_children):
            return 0

        # Find children not yet served (= missing (ivo59apj ?child))
        unserved_children = [c for c in self.goal_children if f"(ivo59apj {c})" not in state]

        # Count trays currently at special location bupp7mhz, as trays are needed to serve sandwiches
        trays_on_special_location = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "y0rxzzmc", "*", self.special_location)
        }
        trays_available = len(trays_on_special_location)
        # Assume trays can be rearranged cheaply, ignore the cost of (whgyhnl0) moves

        # Count all sandwiches currently prepared (i9fq3vit ?yam4s), sandwiches ready on trays (mpvvhjpt ?yam4s ?tray)
        sandwiches_prepared = {
            get_parts(f)[1] for f in state if match(f, "i9fq3vit", "*")
        }
        sandwiches_on_tray = {
            get_parts(f)[1] for f in state if match(f, "mpvvhjpt", "*", "*")
        }
        sandwiches_prepared_or_on_tray = sandwiches_prepared | sandwiches_on_tray

        # Count sandwiches currently unprepared, identified by (zogo0zhl ?yam4s)
        sandwiches_unprepared = {
            get_parts(f)[1] for f in state if match(f, "zogo0zhl", "*")
        }

        # Heuristic calculation per unserved child:
        # For each child:
        # 1) Prepare one sandwich (if no sandwich prepared/unprepared matched, count 1 prep)
        # 2) Place sandwich on tray (1 action)
        # 3) Deliver sandwich to child (1 action)
        #
        # Although a sandwich can be reused? Domain typically models sandwiches as separate objects.
        # We cannot track object bindings well for heuristic, so assume per child sandwich needed.
        #
        # Sandwich preparation can be done by either 'kdzjcqwj' (2-precondition action) or 'h0xkysus' (simpler).
        # We conservatively count only 1 preparation action per child, as heuristic.

        # So total actions per child approx: 3 actions (prep + place on tray + deliver)

        # Now, estimate how many sandwiches are currently prepared or on tray and could be reused:
        # If sandwiches_prepared_or_on_tray < unserved_children:
        #    need to prepare (unserved_children - sandwiches_prepared_or_on_tray) sandwiches

        # To keep heuristic optimistic but informative,
        # count number_of_missing_sandwiches = max(0, unserved_children - prepared_or_on_tray)

        num_prepared_sandwiches = len(sandwiches_prepared_or_on_tray)
        num_unserved = len(unserved_children)

        missing_sandwiches = max(0, num_unserved - num_prepared_sandwiches)

        # Ignore the cost of moving trays; assume trays are ample or can be rearranged (heuristic)

        # Estimate cost components:

        # Cost for preparing missing sandwiches:
        prep_cost = missing_sandwiches * 1  # 1 action per sandwich prep (either kdzjcqwj or h0xkysus)

        # Cost for placing sandwiches on tray:
        # Assume all unserved children need 1 placing action, but sandwiches already on tray don't need placing again.
        # Number of sandwiches on tray may be less than num_unserved, so placing needed for those not on tray.
        num_on_tray = len({
            get_parts(f)[1]
            for f in state
            if match(f, "mpvvhjpt", "*", "*")
        })
        place_cost = max(0, num_unserved - num_on_tray) * 1  # 1 action per placing on tray

        # Cost for delivering sandwiches to children:
        # Each unserved child requires one of (nfarwtet or xtn2vmez) action
        deliver_cost = num_unserved * 1

        total_cost = prep_cost + place_cost + deliver_cost

        # Return heuristic estimate
        return total_cost
