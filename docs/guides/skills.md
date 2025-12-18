# Skills

Skills are modular packages that extend agent capabilities with domain-specific instructions, scripts, and resources. They provide a structured way to give agents expertise in specific areas like data analysis or report writing.

Learn more at [agentskills.io](https://agentskills.io/home)

## Overview

A skill is a directory containing:

- **SKILL.md** - Main instruction file with YAML frontmatter (name, description) and detailed guidance
- **reference/** - Optional subdirectory with reference documentation
- **scripts/** - Optional subdirectory with ready-to-use Python scripts

When skills are loaded, the agent receives:

1. A list of available skills in the system prompt
2. Access to skill files in the execution environment
3. Instructions on how to read and use the skills

## Quick Start

### 1. Create a Skills Directory

```
skills/
└── data_analysis/
    ├── SKILL.md
    ├── reference/
    │   ├── loading.md
    │   └── transformations.md
    └── scripts/
        ├── explore_data.py
        └── summary_stats.py
```

### 2. Create SKILL.md with Frontmatter

```markdown
---
name: data_analysis
description: High-performance data analysis using Polars - load, transform, aggregate, and export tabular data.
---

# Data Analysis Skill

Comprehensive data analysis toolkit using **Polars**.

## Quick Start

```python
import polars as pl
df = pl.read_csv("data.csv")
df.describe()
```

## When to Use This Skill

- Loading datasets (CSV, JSON, Parquet)
- Data cleaning and transformation
- Statistical analysis
...
```

### 3. Pass Skills to the Agent Session

```python
--8<-- "examples/skills/skills_example.py:example"
```

## How Skills Work

When you specify `skills_dir` in `session()`:

1. **Discovery**: Stirrup scans the directory for subdirectories containing `SKILL.md` files
2. **Metadata Extraction**: YAML frontmatter (name, description) is parsed from each `SKILL.md`
3. **System Prompt**: Available skills are listed in the agent's system prompt
4. **File Upload**: The skills directory is uploaded to the execution environment

The agent sees something like this in its system prompt:

```
## Available Skills

You have access to the following skills located in the `skills/` directory.
Each skill contains a SKILL.md file with detailed instructions and potentially bundled scripts.

To use a skill:
1. Read the full instructions: `cat <skill_path>/SKILL.md`
2. Follow the instructions and use any bundled resources as described

- **data_analysis**: High-performance data analysis using Polars (`skills/data_analysis/SKILL.md`)
```

## Creating Effective Skills

### SKILL.md Structure

A well-structured `SKILL.md` should include:

```markdown
---
name: skill_name
description: One-line description shown in the system prompt
---

# Skill Title

Brief overview of what this skill provides.

## Quick Start

Minimal working example the agent can use immediately.

## When to Use This Skill

Bullet list of use cases to help the agent decide when to apply this skill.

## Skill Contents

List reference docs and scripts included in the skill.

## Core Patterns

Common patterns and code snippets for the domain.

## Best Practices

Tips for effective use of the skill.
```

### Including Scripts

Bundle ready-to-use scripts that the agent can execute:

```
skills/data_analysis/scripts/
├── explore_data.py      # Quick dataset profiling
└── summary_stats.py     # Generate statistics report
```

Reference them in SKILL.md:

```markdown
### Ready-to-Use Scripts

- `scripts/explore_data.py` - Quick dataset exploration
  ```bash
  python scripts/explore_data.py data.csv --output report.txt
  ```
```

### Including Reference Documentation

For complex domains, split documentation into focused reference files:

```
skills/data_analysis/reference/
├── loading.md           # Data loading patterns
├── transformations.md   # Column operations
├── aggregations.md      # Group by, window functions
└── visualization.md     # Creating charts
```

Reference them in SKILL.md:

```markdown
### Reference Documentation

- `reference/loading.md` - Loading data from all supported formats
- `reference/transformations.md` - Column operations, filtering, sorting
```

## Example: Data Analysis Skill

See the included example at `skills/data_analysis/`:

```
skills/data_analysis/
├── SKILL.md                          # Main instructions
├── reference/
│   ├── aggregations.md               # Group by, window functions
│   ├── loading.md                    # File format support
│   ├── statistics.md                 # Statistical analysis
│   ├── time_series.md                # Date/time operations
│   ├── transformations.md            # Data transformations
│   └── visualization.md              # Chart creation
└── scripts/
    ├── explore_data.py               # Dataset profiling script
    └── summary_stats.py              # Statistics report script
```

## Skills with Docker

When using skills that require specific dependencies, create a Dockerfile:

```dockerfile
--8<-- "examples/skills/Dockerfile"
```

Then pass it to `DockerCodeExecToolProvider.from_dockerfile()` as shown in the example above.

## Best Practices

1. **Keep skills focused**: Each skill should cover one domain well
2. **Provide working examples**: Quick start code that runs immediately
3. **Include decision guidance**: Help the agent know when to use the skill
4. **Bundle common scripts**: Save agent turns with ready-to-run utilities
5. **Use reference docs for depth**: Keep SKILL.md scannable, put details in reference/

## API Reference

### Session Parameter

```python
agent.session(
    skills_dir="skills",  # Path to skills directory (str or Path)
    ...
)
```

### SkillMetadata

::: stirrup.skills.SkillMetadata
    options:
      show_source: false
      members: false

### Loading Functions

::: stirrup.skills.load_skills_metadata
    options:
      show_source: false

::: stirrup.skills.format_skills_section
    options:
      show_source: false
