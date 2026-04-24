# Implementation Framework

Follow this framework to translate requirements into high-quality code implementation.

## Phase 1: Requirement Analysis
1. **Identify the Core Goal**: What is the primary objective of the request?
2. **Scan for Constraints**: Identify any technical, architectural, or security constraints.
3. **Clarify Ambiguities**: If a requirement is vague, list the possible interpretations and the chosen path.

## Phase 2: Contextual Research
1. **Locate Target Files**: Find the files most relevant to the change.
2. **Identify Dependencies**: What other parts of the system will be affected?
3. **Analyze Existing Patterns**: Study how similar features are implemented in the project.

## Phase 3: Strategy & Design
1. **Draft a Plan**: Outline the steps needed (e.g., Modify API, Update Service, Add Test).
2. **Choose the Surgical Approach**: Aim for the minimal change that fully fulfills the requirement.
3. **Design for Testability**: Ensure the new code can be easily verified with automated tests.

## Phase 4: Execution (Act)
1. **Implement Sub-tasks**: Work through the plan step-by-step.
2. **Adhere to Conventions**: Match the existing style, naming, and architecture.
3. **Maintain Integrity**: Do not skip error handling or type safety for the sake of brevity.

## Phase 5: Validation
1. **Empirical Verification**: Run the code or reproduction script to confirm the fix/feature.
2. **Automated Testing**: Add or update tests to prevent regressions.
3. **Final Review**: Self-review the changes using the [Review Checklist](review-checklist.md).
