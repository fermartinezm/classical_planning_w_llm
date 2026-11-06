# Our Project Explained Simply (Like You’re 5!)

Imagine you have a big pile of LEGO blocks on the floor, and your goal is to stack them into a tall tower in a very specific order: **Red on the bottom, Blue in the middle, Yellow on top.**

How do you figure out the exact steps to get from the messy floor to the finished tower?

That is what this whole project is about: **teaching computers how to make step-by-step plans to solve puzzles**, and testing whether AI chatbots (like ChatGPT or DeepSeek) are actually smart at helping, or just "cheating" because they have seen the puzzles before!

---

## 1. The Big Words Translated into Plain English

Before we look at what our team is doing, let’s translate the fancy computer-science words into everyday ideas:

### 🧩 Classical Planning = "Finding the Steps to Solve a Puzzle"
Imagine a robot in a warehouse, an elevator picking up passengers, or a character pushing boxes in a video game (Sokoban).
* You tell the computer:
  1. **Where everything starts** (e.g., *"Block A is on Block B, and your hand is empty"*).
  2. **What moves are allowed** (e.g., *"You can only pick up a block if nothing is sitting on top of it"*).
  3. **What the goal looks like** (e.g., *"Put Block B on top of Block A"*).
* The computer program that searches for the right sequence of moves is called a **Planner** (in our project, we use a simple Python planner called **Pyperplan** and a fast C++ planner called **Fast Downward**).

### 📜 PDDL = "The Rulebook Language"
Computers don't read English rulebooks well, so scientists invented a simple text format called **PDDL** (Planning Domain Definition Language). It is just a text file that lists the rules of the game (`domain.pddl`) and the starting setup of a specific puzzle (`problem.pddl`).

### 🧭 Heuristic = "A Hint or a ‘Hot-or-Cold’ Meter"
Suppose you are in a giant maze trying to find the exit.
* **Without a hint (Blind Search):** You try every single hallway one by one. If the maze is huge, it takes forever and your computer runs out of time!
* **With a Heuristic (A Hint):** Imagine a friend standing on a ladder who yells a number at you after every step:
  * *"You look about **10 steps** away from the exit!"*
  * If you take a good step, the number goes down to **8** (*"Getting warmer!"*).
  * If you take a bad step, the number goes up to **15** (*"Getting colder!"*).
* A **heuristic** is just a tiny math formula or Python function that looks at the current puzzle board and guesses: **"How many steps are left until the goal?"** Smaller numbers mean *"Go this way!"*

### 🏃‍♂️ Greedy Best-First Search (GBFS) = "Always Follow the ‘Warmest’ Hint"
When our planner uses **Greedy Best-First Search**, it simply looks at all the moves it could make next, asks the heuristic (the hint-giver) to score each one, and **always jumps to the board with the smallest score** (the one that looks closest to the goal).

### 🤖 `h_FF` = "The Famous Classic Hint-Giver"
For 25 years, scientists have built general-purpose hint-givers that work on *any* puzzle. The most famous one is called **`h_FF`**. It is our **measuring stick (baseline)**. If a new hint-giver solves more puzzles than `h_FF`, that’s a big deal!

### 📏 Benchmark = "A Standardized Obstacle Course"
To compare hint-givers fairly, everyone runs them on the exact same set of puzzles with the same stopwatch limit (5 minutes per puzzle). Our obstacle course has **8 different games (domains)**—like stacking blocks (*Blocksworld*), driving rovers on Mars (*Rovers*), making sandwiches (*Childsnack*), and pushing crates (*Sokoban*)—with **30 test puzzles per game (240 puzzles total)**.
* **Coverage** just means: *"Out of 240 puzzles, how many did the computer finish before the 5-minute buzzer rang?"*

---

## 2. What Did the Original Paper Discover? (The Backstory)

Earlier in 2025, a group of researchers (Corrêa et al.) tried something clever:
1. Remember that AI chatbots (LLMs like ChatGPT or DeepSeek) are actually terrible at solving a 50-step puzzle directly—they lose track and make illegal moves.
2. Instead of asking the AI to solve every puzzle, the researchers said:
   > *"Hey AI, here are the rules of the game (the PDDL file). Write me a short **Python hint function (a heuristic)** for this game! Then our regular planner will use your hint function to solve all the puzzles."*
3. They asked the AI to write **25 different hint functions** for each game, tested all 25 on some easy practice puzzles (**training tasks**), picked the #1 winner, and used it on the hard **test puzzles**.
4. **Their big headline:** The AI-written Python hints beat the famous classic hint (`h_FF`)!

---

## 3. What Are WE Doing, and Why Does It Matter?

When a scientist sees a headline like *"AI writes better hints than classic algorithms!"*, the very next question is:
> **"Wait a second... WHY did it work? Is the AI genuinely understanding the logic of the puzzle, or did it just get lucky?"**

Our team of 4 is putting that paper under a microscope across **three big questions (Axes)**:

### 🕵️‍♀️ Question 1 (Member 1 — Obfuscation): "Is the AI just recognizing famous games from memory?"
* **The suspicion:** Games like *Blocksworld* and *Sokoban* are in every textbook on the internet. When the AI sees words like `pick-up`, `stack`, `on-table`, or `push-stone`, it immediately knows what game it is and recites a textbook formula from memory—without actually reading the rules!
* **How we test it:** We disguise the words in the rulebook at 4 levels:
  * **L0 (Original):** Normal words (`on-table`, `stack`).
  * **L1 (Secret Code Names):** Replace words with gibberish (`pred_1`, `action_4`).
  * **L2 (Gibberish + Scrambled Order):** Scramble the order of the rules too.
  * **L3 (Trick / Misleading Words):** Swap the words of a block-stacking game with words from a truck-driving game (so stacking a block is called `drive-truck`)!
* **Why it’s cool:** Classic `h_FF` doesn't care what words you use—it only reads the math. If the AI's score crashes when we hide or swap the words, it proves the AI relies on English word clues rather than pure logic!

### 🐣 Question 2 (Member 2 — Model Scale): "Do you need a giant, expensive supercomputer AI, or can a tiny laptop AI do this?"
* **The suspicion:** The original paper used **DeepSeek-R1**, a massive 671-billion-parameter model running on giant servers. What happens if we use much smaller, free AI models (like 3B, 7B, and 14B models) that can run on a normal laptop?
* **How we test it:** We compare the giant models against small models, and we also track **how** small models fail:
  * Do they write broken Python code that crashes (`crash_compile` / `crash_runtime`)?
  * Do they write a hint that says *"Infinity steps left!"* on every move (`infinite_h`)?
  * Or do they write working code that just gives bad advice and times out (`timeout`)?

### 🎯 Question 3 (Member 3 — Selection & Teamwork): "Are we picking the wrong hint out of the 25 tries?"
* **The suspicion:** When the AI writes 25 candidate hints for a game, some are terrible and a few are brilliant. The original paper picked whichever candidate solved the most *easy training puzzles*. What if 10 candidates all solve 100% of the easy puzzles, and the paper picked a mediocre one by accident?
* **How we test it:**
  1. We test all 25 already-saved hints on the test puzzles to see what the **perfect pick (the Oracle)** would have scored, and test smarter ways of picking a winner (like only looking at the hardest training puzzles, or picking the hint that takes the fewest steps).
  2. **Teamwork (Alternation):** Instead of picking just **1** winner, what if we take the **top 3 hints** and have the planner take turns listening to Hint #1, then Hint #2, then Hint #3 on every step? (This "round-robin" trick is how champion planners like LAMA work!)

### 🏗️ Your Role (Member 4 — The Referee, Track Builder & Storyteller)
You are the backbone of the team:
1. **You built the shared race track (`run_experiments.py` and `benchmarks/splits.json`):** Everyone runs their puzzles through your script so the stopwatch (5 minutes), memory limit (4 GB), and scorecards (`.jsonl` files) are 100% fair and identical.
2. **You run the measuring sticks (Baselines):** You run the classic planners (**Blind Search**, **Pyperplan `h_FF`**, **Fast Downward `h_FF`**, **LAMA**) and re-run the **Original Paper’s AI hints** under the 5-minute clock so Members 1, 2, and 3 have a fair baseline to compare against.
3. **You generate the big-picture charts (`plots/make_plots_and_tables.py`) and frame the paper:**
   * **Figure F4 (Scatter Plot):** Does the AI hint actually explore fewer dead-end rooms in the maze than `h_FF`, or is it just faster to calculate?
   * **Figure F5 (Cactus Plot):** One grand summary race chart showing how many puzzles each method solves after 1 second, 10 seconds, 1 minute, and 5 minutes.

---

## 4. What Kind of Results Can We Get? (Why You Can't Lose!)

Because this is a **benchmarking (stress-testing) project**, **every possible outcome is a great scientific result**:

* **What we expect to see in Axis 1 (Disguised Words):**
  * When we replace real words with gibberish (L1/L2), the AI's puzzles solved will drop noticeably—especially on famous textbook games like *Blocksworld* and *Sokoban*.
  * When we use **misleading words (L3)** (calling blocks "trucks"), the AI may get actively tricked into writing a truck-driving formula for a block puzzle, performing even worse than gibberish! Meanwhile, `h_FF` will get the exact same score every time.
* **What we expect to see in Axis 2 (Small vs. Giant Models):**
  * Giant models (DeepSeek-R1/V3) beat `h_FF`, mid-sized models (14B) fall just below `h_FF`, and very small models (3B/7B) will show a clear spike in code crashes or "infinity" bugs. Even if small models fail completely, **showing the exact size where AI hints break down (and why they crash)** is the exact finding Figure F2 is designed to show!
* **What we expect to see in Axis 3 (Picking & Combining Hints):**
  * We will likely find a gap between the hint the paper picked on training puzzles and the actual best hint among the 25 candidates—and we expect **combining the top 3 hints (Alternation)** to beat any single pick because if one hint gets stuck in a dead end, the other two rescue it!
* **What we expect to see in Your Charts (Member 4 — F4 & F5):**
  * On puzzles where the AI hint wins, your scatter plot (**F4**) will show dots below the diagonal line—meaning the AI hint genuinely guides the planner through **10× to 100× fewer dead ends** than `h_FF`.
  * At the same time, your cactus plot (**F5**) will show that while AI hints make a slow Python planner (Pyperplan) much smarter, a modern C++ engine (Fast Downward / LAMA) is still way faster in raw speed—giving your report a balanced, honest conclusion!

