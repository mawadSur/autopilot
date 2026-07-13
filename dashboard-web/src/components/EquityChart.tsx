"use client";

import { useEffect, useRef } from "react";
import type { EquityPoint } from "@/lib/types";
import { fmtUsd } from "@/lib/format";

// Hand-rolled canvas equity curve, ported from the original dashboard's
// drawChart(). Redraws on data or resize; animates the draw once per change.

const C = {
  grid: "#1f2940",
  axis: "#6b7c99",
  base: "#2e3b57",
  label: "#f1f5f9",
  green: "#22d88f",
  red: "#f4515f",
  greenFill: "rgba(34,216,143,.20)",
  redFill: "rgba(244,81,95,.18)",
};
const CHART_FONT = "11px 'Fira Code', ui-monospace, monospace";

function draw(canvas: HTMLCanvasElement, curve: EquityPoint[], progress: number) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 600;
  const cssH = 220;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  if (!curve || curve.length < 2) return;

  const padL = 58, padR = 16, padT = 14, padB = 26;
  const w = cssW - padL - padR, h = cssH - padT - padB;
  const ys = curve.map((p) => p.equity);
  let min = Math.min(...ys), max = Math.max(...ys);
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  min -= span * 0.08; max += span * 0.08;

  const x = (i: number) => padL + (w * i) / (curve.length - 1);
  const y = (v: number) => padT + h - (h * (v - min)) / (max - min);

  ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
  ctx.fillStyle = C.axis; ctx.font = CHART_FONT; ctx.textAlign = "right";
  const rows = 4;
  for (let r = 0; r <= rows; r++) {
    const v = min + ((max - min) * r) / rows;
    const yy = y(v);
    ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(cssW - padR, yy); ctx.stroke();
    ctx.fillText("$" + v.toFixed(0), padL - 8, yy + 3);
  }

  const base = curve[0].equity;
  ctx.save();
  ctx.setLineDash([4, 4]); ctx.strokeStyle = C.base; ctx.beginPath();
  ctx.moveTo(padL, y(base)); ctx.lineTo(cssW - padR, y(base)); ctx.stroke();
  ctx.restore();

  const total = curve.length - 1;
  const shown = Math.max(1, Math.floor(total * progress));
  const frac = total * progress - shown;
  const last = curve[curve.length - 1].equity;
  const lineColor = last >= base ? C.green : C.red;

  ctx.beginPath();
  ctx.moveTo(x(0), y(curve[0].equity));
  for (let i = 1; i <= shown; i++) ctx.lineTo(x(i), y(curve[i].equity));
  let endX = x(shown), endY = y(curve[shown] ? curve[shown].equity : curve[shown - 1].equity);
  if (frac > 0 && shown < total) {
    const a = curve[shown], b = curve[shown + 1];
    endX = x(shown) + (x(shown + 1) - x(shown)) * frac;
    endY = y(a.equity + (b.equity - a.equity) * frac);
    ctx.lineTo(endX, endY);
  }
  const grad = ctx.createLinearGradient(0, padT, 0, padT + h);
  grad.addColorStop(0, last >= base ? C.greenFill : C.redFill);
  grad.addColorStop(1, "rgba(0,0,0,0)");
  ctx.lineTo(endX, padT + h); ctx.lineTo(x(0), padT + h); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();

  ctx.beginPath();
  ctx.moveTo(x(0), y(curve[0].equity));
  for (let i = 1; i <= shown; i++) ctx.lineTo(x(i), y(curve[i].equity));
  if (frac > 0 && shown < total) ctx.lineTo(endX, endY);
  ctx.strokeStyle = lineColor; ctx.lineWidth = 2.2; ctx.lineJoin = "round";
  ctx.shadowColor = lineColor; ctx.shadowBlur = 9; ctx.stroke(); ctx.shadowBlur = 0;

  if (progress >= 1) {
    ctx.beginPath(); ctx.arc(x(total), y(last), 3.6, 0, Math.PI * 2);
    ctx.fillStyle = lineColor; ctx.fill();
    ctx.fillStyle = C.label; ctx.textAlign = "right";
    ctx.font = "12px 'Fira Code', ui-monospace, monospace";
    const ly = Math.min(Math.max(y(last) - 8, padT + 10), padT + h - 4);
    ctx.fillText("$" + last.toFixed(2), cssW - padR, ly);
  }
}

export default function EquityChart({ curve }: { curve: EquityPoint[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const lastKeyRef = useRef<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const reduce =
      typeof window !== "undefined" &&
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const key = JSON.stringify(curve.map((p) => [p.t, Math.round(p.equity * 100)]));
    const animate = key !== lastKeyRef.current && !reduce;
    lastKeyRef.current = key;

    if (!animate) { draw(canvas, curve, 1); return; }
    const start = performance.now(), dur = 750;
    const step = (now: number) => {
      const p = Math.min(1, (now - start) / dur);
      draw(canvas, curve, p);
      if (p < 1) rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [curve]);

  useEffect(() => {
    const onResize = () => {
      const canvas = canvasRef.current;
      if (canvas) draw(canvas, curve, 1);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [curve]);

  const last = curve.length ? curve[curve.length - 1].equity : null;
  const hasCurve = curve && curve.length >= 2;

  return (
    <div className="panel">
      <h2>
        Equity curve <span className="count mono">{fmtUsd(last)}</span>
      </h2>
      <div id="chartWrap">
        {hasCurve ? (
          <canvas ref={canvasRef} height={220} />
        ) : (
          <div className="chart-empty">
            No settled trades yet — the equity curve appears once positions resolve.
          </div>
        )}
      </div>
    </div>
  );
}
