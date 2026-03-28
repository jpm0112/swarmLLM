# Website — Claude Instructions

This directory contains the SwarmLLM project website (React + Vite, deployed to GitHub Pages).

## Scope

**Do NOT modify any file outside of `website/` without explicit instructions from the user.** This includes:
- Root-level files (`README.md`, `pyproject.toml`, etc.)
- Source package (`swarmllm/`)
- Tests, scripts, configs, docs
- GitHub Actions workflows (`.github/`)

All work should be confined to this `website/` directory.

## Project Context

SwarmLLM is a research prototype for coordinator-guided LLM swarm optimization. A central coordinator LLM directs multiple worker LLM agents to explore solution spaces for combinatorial optimization problems (job scheduling, job shop scheduling). Agents generate Python heuristics, execute them in a sandbox, and share results via inspectable markdown logs.

**Collaborators (equal contribution):**
- Konstantinos Ziliaskopoulos
- Juan Pablo Morande

## Design Philosophy

- Bold, opinionated, technical aesthetic — this is a research/CV project
- Dark-first design with high contrast and strong typography
- Animated or interactive elements are encouraged to stand out
- Must be informative: architecture, features, getting started, collaborators
- Target audience: researchers, ML engineers, potential employers/collaborators

## Tech Stack

- React 19 + Vite
- Plain CSS (no Tailwind unless added intentionally)
- Deployed via GitHub Pages (output: `dist/`)
