# Code Review Checklist

Use this checklist to ensure high-quality, maintainable, and secure code changes.

## 1. Correctness & Requirements
- [ ] Does the code accurately implement the requested feature or fix?
- [ ] Are there any edge cases (nulls, empty inputs, network errors) not handled?
- [ ] Does the implementation match the project's architectural requirements?

## 2. Readability & Maintainability
- [ ] Is the code easy to understand? Are variable and function names descriptive?
- [ ] Is there redundant code that could be refactored or simplified?
- [ ] Are complex sections documented with clear, purposeful comments?
- [ ] Does it follow the established naming and formatting conventions?

## 3. Architecture & Design
- [ ] Does the change fit within the existing project structure?
- [ ] Are concerns properly separated (e.g., business logic vs. API handling)?
- [ ] Does it introduce unnecessary dependencies or "just-in-case" logic?

## 4. Performance & Efficiency
- [ ] Are there any obvious performance bottlenecks (e.g., N+1 queries, inefficient loops)?
- [ ] Is resource usage (memory, disk, network) optimized for the context?

## 5. Security & Safety
- [ ] Are sensitive credentials, API keys, or secrets protected (never logged or committed)?
- [ ] Is user input properly validated and sanitized?
- [ ] Are there any potential race conditions or thread-safety issues?

## 6. Testability
- [ ] Is the code designed to be easily testable?
- [ ] Are there sufficient tests (unit, integration) covering the new logic?
- [ ] Do existing tests still pass?
