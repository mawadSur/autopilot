---
name: project-planning
description: >
  Master project planning skill. Use this skill whenever the user wants to plan
  a project, break down tasks, create a roadmap, generate a todo list, define
  milestones, prioritize features, or structure a development workflow.
  Works with Gemini CLI /plan mode, Claude Code, and Codex CLI.
platforms: [gemini-cli, claude-code, codex-cli]
---

# Project Planning Skill

## Skill Dependency Map

  Task Type                              Skill to Activate
  ---------------------------------------------------------------
  Project setup / config                 firebase-basics
  Authentication / login                 firebase-auth-basics
  Firestore DB standard                  firebase-firestore-standard
  Firestore DB enterprise                firebase-firestore-enterprise-native-mode
  Security rules review                  firestore-security-rules-auditor
  Static site hosting                    firebase-hosting-basics
  Full-stack SSR deployment              firebase-app-hosting-basics
  SQL / GraphQL data layer               firebase-data-connect
  AI features with Gemini SDK            firebase-ai-logic
  AI flows in JavaScript/TypeScript      developing-genkit-js
  AI flows in Go                         developing-genkit-go
  AI in Flutter/Dart                     developing-genkit-dart

## Planning Phases

  Phase 1 - Setup        firebase-basics, firebase-auth-basics
  Phase 2 - Core         firebase-firestore-standard, firebase-data-connect
  Phase 3 - Enhancement  firebase-ai-logic, developing-genkit-js
  Phase 4 - Security     firestore-security-rules-auditor
  Phase 5 - Deploy       firebase-hosting-basics, firebase-app-hosting-basics

## Task Format

  [ ] TASK-001  Task title
                Priority: High | Medium | Low
                Skill:    skill-name-here
                Effort:   Small (<1hr) | Medium (1-4hr) | Large (4hr+)

## Platform Instructions

  Gemini CLI:   /plan then say "plan my project"
                /todo add "task" to register tasks
                activate_skill <name> when starting each task

  Claude Code:  Add to CLAUDE.md:
                ## Plan
                - [ ] Task (skill-name)

  Codex CLI:    Step 1: [.codex/skills/firebase-basics] Initialize project
                Step 2: [.codex/skills/firebase-auth-basics] Add auth

## Trigger Phrases
  "plan this project" | "help me plan" | "create a roadmap"
  "break this down" | "what do I build first" | "make a todo list"
