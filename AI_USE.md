# AI-use disclosure

## Tool
Claude Code (Anthropic's coding agent CLI), run in this repository.

## What it was used for
- Creating the project skeleton (folders, config, Makefile, utility module, tests).
- Writing and debugging pipeline code: data download, splits, feature
  extraction, model training, evaluation and the website.
- Drafting documentation: README, DECISIONS.md and the study notes in `learn/`.

## What it was not used for
- Choosing the research question, the main split or the evaluation metric. Those
  choices are mine and are justified in DECISIONS.md.
- Producing any number in the report. Every reported number comes from running
  the code in this repository, never from text the model wrote.

## How outputs are checked
- **Tests:** pytest tests cover splits, windowing, features and evaluation;
  `make tests` must pass before each commit.
- **Running it:** every script is run end to end; results are read from the files
  it writes (`results/`), not from the agent's summary.
- **Leakage guards:** the test set is locked in code until the git tag
  `pre-test-freeze` exists; preprocessing is fit on training data only.
- **Review:** I read every diff before committing and must be able to explain
  each design choice (see `learn/` for my notes).
- **History:** git commits record what changed at each step.
