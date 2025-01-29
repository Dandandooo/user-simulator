# Simulating User Agents for Embodied Conversational AI
Daniel Philipov, Vardhan Dongre, Gokhan Tur, Dilek Hakkani-Tür
([arxiv.org](https://arxiv.org/abs/2410.23535))

## Abstract
Embodied agents designed to assist users with
tasks must possess the ability to engage in natu-
ral language interactions, interpret user instruc-
tions, execute actions to complete tasks, and
communicate effectively to resolve any issues
that arise. Building such agents require datasets
which are expensive and labor intensive to col-
lect. Further interactive datasets are needed to
evaluate them. In this work, we propose build-
ing an LLM-based user proxy agent that simu-
lates user behavior in a human-robot interaction
using a virtual environment for embodied Con-
versational AI. Given a concrete user goal (e.g.,
make breakfast), at each time step during the
interaction, the user agent may "observe" the
robot actions or "converse" in order to either
proactively intervene with the robot behavior
or reactively respond to the robot when needed.
Such a user agent would be critical for improv-
ing and evaluating the robot interaction and
task completion ability, and for potential future
research, such as reinforcement learning using
AI feedback. We assess the user agent’s abil-
ity to mimic the actual user actions using the
TEACh dataset, demonstrating the feasibility
of our approach to enhance the effectiveness
and reliability of human-robot interactions in
achieving tasks through natural language com-
munication.

