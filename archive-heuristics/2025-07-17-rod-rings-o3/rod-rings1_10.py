import math
from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """
    Utility helper:  "(on r2 r3)"  ->  ["on", "r2", "r3"]
    """
    return fact[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A simple, fast, domain–dependent heuristic for the *rod-rings* domain.
    The heuristic counts how many goal facts of each type (top / bot / on /
    others) are still unsatisfied in the current state and converts this
    information into an estimated number of additional actions that will be
    required to reach a goal state.

    Assumptions
    -----------
    1. Exactly one ring is always held; the goals never restrict the *holding*
       predicate, therefore it can be ignored.
    2. Every action (`put_top` or `put_bottom`)                                     
       • makes exactly **one** ring become *top* **or** *bot* on some stick, and  
       • establishes **one** `on` relationship that involves that ring.           
       Consequently, an action can satisfy at most                                 
         – one `top`/`bot` goal **and**                                           
         – one `on` goal.                                                          
    3. No single action can satisfy more than one `top`/`bot` goal or more than
       one `on` goal, and it never satisfies two unrelated goal facts.

    Heuristic Initialization
    ------------------------
    During construction the heuristic only stores the set of goal facts, because
    the domain supplies no static facts that influence the estimate.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1.  Determine the set **G_unsat** of goal facts that are not contained in
        the current state.
    2.  Partition **G_unsat** into                              
            • `unsat_top_bot` : number of `(top …)` *plus* `(bot …)` facts,      
            • `unsat_on`      : number of `(on … …)` facts,                      
            • `unsat_other`   : everything else (normally 0 in this domain).     
    3.  Each `top`/`bot` fact needs **at least one** action (because only the
        ring that becomes extremal can satisfy it).  The same is true for each
        `on` fact; however one action that fixes a `top`/`bot` fact can *also*
        fix *one* `on` fact at the same time.                                    
        Hence                                                                               
              actions ≥ max(unsat_top_bot, unsat_on).                                      
    4.  Add the goals of other predicates individually because no action is able
        to satisfy them “for free”.  The final estimate therefore is            

            heuristic(state)  =  unsat_other  +  max(unsat_top_bot, unsat_on)

    The value is 0 in every goal state and finite for any reachable state.  It
    is *inadmissible* (may over-estimate) but quick to compute and still
    informative enough to guide a greedy best-first search effectively.
    """

    def __init__(self, task):
        # Store the (immutable) set of goal facts for quick difference tests
        self.goals = task.goals

    def __call__(self, node):
        state = node.state

        # ---------------------------------------------------------------------
        # 1.  Unsatisfied goal facts
        # ---------------------------------------------------------------------
        unsatisfied = self.goals - state
        if not unsatisfied:
            return 0  # Goal reached

        # ---------------------------------------------------------------------
        # 2.  Count how many are of each relevant predicate
        # ---------------------------------------------------------------------
        unsat_top_bot = 0
        unsat_on = 0
        for fact in unsatisfied:
            predicate = get_parts(fact)[0]
            if predicate in ("top", "bot"):
                unsat_top_bot += 1
            elif predicate == "on":
                unsat_on += 1

        unsat_other = len(unsatisfied) - unsat_top_bot - unsat_on

        # ---------------------------------------------------------------------
        # 3.  Heuristic computation   (see docstring for rationale)
        # ---------------------------------------------------------------------
        estimated_actions = unsat_other + max(unsat_top_bot, unsat_on)

        return estimated_actions
