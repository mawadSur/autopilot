import { fileURLToPath } from "node:url";
import { dirname } from "node:path";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The dashboard reads bot state from Vercel KV at request time; never cache
  // API responses at the framework layer.
  poweredByHeader: false,
  // The repo root has its own package-lock.json (the Python project's
  // @sentry/node). Pin the tracing root to this app so Next doesn't infer the
  // monorepo root and mis-trace the standalone bundle on Vercel.
  outputFileTracingRoot: dirname(fileURLToPath(import.meta.url)),
};

export default nextConfig;
