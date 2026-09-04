# Communication and Style

- Be concise, precise, direct, information-dense, and technical.
- Do not use emojis, em-dashes, fluff, filler, or marketing language.
- Avoid fancy bullet formatting such as bold prefixes. Use simple, plain bullet points.
- Use inclusive, modern terminology at all times.

# Repository Management

- Keep this file lean. Only add new guidelines or project instructions with explicit user permission.
- Keep .gitignore minimal. Only add entries when required and do not bloat it.
- Use conventional commits for all git commit messages.

# Codebase Architecture

- Maintain a modular structure with clear separation of concerns.
- Isolate configuration, data pipelines, core logic, and experiment scripts.
- Write readable, maintainable, and reusable code with clean interfaces.

# Performance and Portability

- Write platform-agnostic code compatible across macOS, Linux, and Windows.
- Dynamically detect and leverage available hardware accelerators (CUDA, MPS, CPU).
- Maximize performance by parallelizing workloads across available CPU and GPU resources.

# Environment and Dependencies

- Use the uv skill for all environment, package, and script execution workflows.
- Target Python 3.14 by default. Downgrade only when strictly required for package compatibility.
- Do not install packages or system tools (such as uv add, brew, or apt) without explicit prior user approval.

# Code Quality and Type Safety

- Use the ruff skill for formatting and linting.
- Use the ty skill for type checking.
- Provide explicit static type annotations on all function signatures and module interfaces.

# Experiment Reproducibility

- Ensure deterministic experiment runs by setting random seeds across torch, numpy, and random.
