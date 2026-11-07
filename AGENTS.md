# Repository Guidelines

## Project Status and Layout

This IIT Gandhinagar Multi-Agent Systems project is at the idea-review stage. `README.md` introduces the project. `FAI_Project_Idea_Review.md` describes the research question, model assumptions, baselines, and proposed evaluation. Use it as the starting point for implementation decisions.

## Editing the Idea Review

Use descriptive Markdown headings and follow the proposal’s notation. When changing an assumption or definition, check the related baselines, metrics, and experiment plan for consistency. Label hypotheses and proposed experiments explicitly. When reporting results, record the setup and evidence behind each claim.

## Development and Verification

The current checkout contains documentation. For documentation edits, run `git diff --check` before staging and preview changed Markdown. Check links, equations, and tables. For newly staged files, run `git diff --cached --check`.

Add source and test directories when the first runnable code lands. Document dependency installation, a minimal simulator run, and the test command in the README alongside that change. Add tests for the model behavior the change introduces and record any seeds or parameters needed to reproduce experiment outputs.

## Commits and Pull Requests

Use short, imperative commit subjects such as `Clarify the audit assumptions`. Keep each pull request focused on one change. Describe the decision or research question it addresses, list what you checked, and link the issue if one exists. Keep unrelated coursework and generated exports out of project commits.
