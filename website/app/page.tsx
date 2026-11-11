import { ChartFigure } from "./ChartFigure";
import { Card, Kpi } from "./components";

const objectives = [
  { title: "Baseline Benchmark", text: "Run 89 tasks on 3 harnesses, same model; record results" },
  {
    title: "Analyse",
    text: "Find where harnesses disagree and which complete each task for the lowest cost",
  },
  { title: "Route", text: "Rule-based vs LLM-classifier router, plus one fallback retry" },
  { title: "Evaluate", text: "Accuracy, tokens and cost vs the best single harness" },
];

const harnesses = ["terminus-2", "mini-swe-agent", "pi"];

const capabilities = [
  { icon: "✓", tone: "cap-ok", text: "Pass → return result" },
  { icon: "↻", tone: "cap-retry", text: "Harness error → try 2nd-best choice" },
  { icon: "↻", tone: "cap-retry", text: "Routing error → try single-best" },
  { icon: "$", tone: "cap-cost", text: "Log tokens & cost" },
];

const stages = [
  { title: "Baseline & Analysis", text: "89 tasks × 3 harnesses: pass/fail, tokens, cost" },
  {
    title: "Offline router dev",
    text: "Compare both routing strategies on saved results, no re-runs",
  },
  {
    title: "Final run",
    text: "Meta-router using best routing strategy on all 89 tasks, scored against single best harnesses",
  },
];

const takeaways = [
  "Routing edges out the best harness: 29/89 vs 28/89 (+1.1 pp)",
  "Baseline run results suggest live-terminal tasks suit Terminus-2; most others suit mini-swe-agent",
  "Marginal gain: +1 task and 0.7% lower cost; 36/89 theoretical ceiling unmet",
];

const impact = [
  "Lower cost per solved task",
  "Per-task harness dataset",
  "Reproducible offline evaluation",
  "Evidence on meta-layer overhead",
];

export default function Home() {
  return (
    <>
      <div className="backdrop" aria-hidden="true">
        <img src="/img/background.svg" alt="" />
      </div>

      <main className="page">
        <header className="hero">
          <h1 className="title">Harness Router for Terminal-Bench 2.1</h1>
          <p className="subtitle">
            Dispatch each Terminal-Bench 2.1 task to the harness most likely to solve it, with the
            model held constant, to beat the best single harness or match it at lower cost.
          </p>
        </header>

        <div className="row row-1">
          <Card id="problem" number="01" title="Problem & Motivation">
            <div className="stack">
              <div className="tile">
                <p className="label label-accent">The problem</p>
                <p className="body-text">
                  Every agent operates within a harness: the software managing its tools, context,
                  feedback and stopping conditions. Holding the model fixed and changing the harness
                  can affect task performance and cost.
                </p>
                <div className="flow-chips">
                  <span className="chip">Same model</span>
                  <span className="arrow">→</span>
                  <span className="chip chip-accent">3 harnesses</span>
                  <span className="arrow">→</span>
                  <span className="chip">No harness wins every task</span>
                </div>
              </div>
              <div className="kpis">
                <Kpi
                  value="36/89"
                  label="Theoretical meta-harness ceiling, based on baseline runs of selected harnesses"
                />
                <Kpi value="1.9×" label="cost gap across harnesses" />
                <Kpi value="40×" label="tokens per solve, same model" note="Vats & Golev, 2026" />
              </div>
            </div>
          </Card>

          <Card id="objectives" number="02" title="Project Objectives">
            <ol className="objectives">
              {objectives.map((o, i) => (
                <li key={o.title} className="objective">
                  <span className="step">{String(i + 1).padStart(2, "0")}</span>
                  <div>
                    <h3>{o.title}</h3>
                    <p>{o.text}</p>
                  </div>
                </li>
              ))}
            </ol>
          </Card>

          <Card id="approach" number="03" title="Approach & Development Stages">
            <div className="stack">
              <p className="label">Routing pipeline</p>
              <div className="pipeline">
                <div className="node">Task</div>
                <span className="pipe-arrow" aria-hidden="true">→</span>
                <div className="node node-router">
                  Meta-Router
                  <span>Built on Harbor’s native classes</span>
                  <span>Strategy: rule-based | LLM classifier</span>
                </div>
                <span className="pipe-arrow" aria-hidden="true">→</span>
                <ul className="harness-pool" aria-label="Harness pool">
                  {harnesses.map((h) => (
                    <li key={h} className="harness">
                      {h}
                    </li>
                  ))}
                </ul>
                <span className="pipe-arrow" aria-hidden="true">→</span>
                <div className="node verify">
                  <p className="label">Verify &amp; fallback</p>
                  {capabilities.map((c) => (
                    <div key={c.text} className="capability">
                      <span className={`cap-icon ${c.tone}`}>{c.icon}</span>
                      {c.text}
                    </div>
                  ))}
                </div>
              </div>
              <ol className="stages">
                {stages.map((s, i) => (
                  <li key={s.title} className="stage">
                    <p className="label label-accent">Stage {i + 1}</p>
                    <h3>{s.title}</h3>
                    <p>{s.text}</p>
                  </li>
                ))}
              </ol>
            </div>
          </Card>
        </div>

        <div className="row row-2">
          <Card
            id="results"
            number="04"
            title="Key Results & Findings: Meta-Router, Rules Based Strategy"
          >
            <div className="results">
              <div className="kpis">
                <Kpi value="33%" label="Router pass rate (29/89 tasks)" />
                <Kpi value="+1.1 pp" label="pass rate vs single best" />
                <Kpi value="−0.7%" label="estimated model API cost" />
              </div>
              <ChartFigure
                src="/img/results-table.png"
                alt="Per-task pass/fail matrix across harnesses and the oracle"
              />
              <ChartFigure
                src="/img/success-rate.png"
                alt="Task success rate by configuration (successful tasks / 89)"
              />
              <ChartFigure
                src="/img/accuracy-cost.png"
                alt="Accuracy vs total model cost, with the Pareto front highlighted"
              />
            </div>
          </Card>

          <Card id="conclusions" number="05" title="Conclusions & Project Impact">
            <div className="stack">
              <ul className="takeaways">
                {takeaways.map((t) => (
                  <li key={t} className="takeaway">
                    <span className="check" aria-hidden="true">✓</span>
                    {t}
                  </li>
                ))}
              </ul>
              <p className="label">Impact</p>
              <ul className="impact">
                {impact.map((t) => (
                  <li key={t}>{t}</li>
                ))}
              </ul>
            </div>
          </Card>
        </div>

        <footer className="footer">
          <div className="logos">
            <img src="/img/logo-uts.png" alt="UTS" />
            <img src="/img/logo-synogize.png" alt="Synogize" />
          </div>
        </footer>
      </main>
    </>
  );
}
