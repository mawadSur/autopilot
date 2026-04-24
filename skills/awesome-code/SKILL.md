---
name: awesome-code
description: Advanced code assistant for comprehensive code reviews, requirement analysis, and high-quality implementation. Use when you need to ensure code meets standards, matches project requirements, or requires complex refactoring and feature implementation.
---

# Awesome Code

This skill transforms Gemini CLI into a senior-level code reviewer and implementation specialist. It provides a structured approach to analyzing requirements, reviewing existing code, and executing surgical, high-quality changes.

## Core Capabilities

### 1. Requirements Understanding
Deeply analyze user requests to extract intent, constraints, and success criteria.
- **Workflow**: See [Implementation Framework](references/implementation-framework.md) Phase 1.

### 2. Comprehensive Code Review
Evaluate code for correctness, maintainability, performance, and security.
- **Workflow**: Use the [Code Review Checklist](references/review-checklist.md) for every review task.

### 3. Surgical Implementation
Execute changes that are idiomatic, complete, and minimally invasive.
- **Workflow**: Follow the Research -> Strategy -> Execution lifecycle detailed in the [Implementation Framework](references/implementation-framework.md).

## Usage Guide

### Reviewing Code
When asked to review code or a pull request:
1. Load the [Code Review Checklist](references/review-checklist.md).
2. Categorize feedback into Correctness, Readability, Architecture, Performance, and Security.
3. Provide actionable suggestions with code examples.

### Implementing Features/Fixes
When a direct instruction to modify code is given:
1. Apply the [Implementation Framework](references/implementation-framework.md).
2. Start with a "Requirement Analysis" summary to align with the user.
3. Propose a "Strategy" before acting.
4. Validate every change with tests.

### Refactoring
When asked to improve existing code:
1. Identify smells using the [Code Review Checklist](references/review-checklist.md).
2. Plan a multi-step refactor that preserves behavior while improving structure.
3. Verify with existing tests at each step.
