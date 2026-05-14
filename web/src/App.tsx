import {
  Activity,
  AlertTriangle,
  BarChart3,
  ClipboardCheck,
  Database,
  Gauge,
  Info,
  Layers3,
  Shuffle,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  type LucideIcon,
} from "lucide-react";
import { type FormEvent, useEffect, useMemo, useState } from "react";
import {
  FeatureMeta,
  FeaturesResponse,
  getBasicFeatures,
  getMetrics,
  getTransactions,
  MetricsResponse,
  ModelKey,
  predictBasic,
  PredictionResponse,
  ThresholdRow,
  TransactionQuery,
  TransactionRow,
  TransactionsResponse,
  ViewKey,
} from "./api";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Line,
  LineChart,
  ResponsiveContainer,
  Sankey,
  type SankeyLinkProps,
  type SankeyNodeProps,
  Tooltip,
  useChartWidth,
  XAxis,
  YAxis,
} from "recharts";

const views: Array<{ key: ViewKey; label: string; icon: LucideIcon }> = [
  { key: "overview", label: "Overview", icon: Gauge },
  { key: "transactions", label: "Transactions", icon: Search },
  { key: "thresholds", label: "Thresholds", icon: SlidersHorizontal },
  { key: "custom", label: "Custom Check", icon: ClipboardCheck },
  { key: "data", label: "About", icon: Info },
];

const TRANSACTION_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

function formatNumber(value: number | undefined, maximumFractionDigits = 0) {
  if (value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("en-GB", { maximumFractionDigits }).format(value);
}

function formatPercent(value: number | undefined, digits = 1) {
  if (value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function probabilityPercent(value: number) {
  return Math.round(value * 1000) / 10;
}

function classForRisk(value: number) {
  if (value >= 0.5) return "risk-high";
  if (value >= 0.3) return "risk-elevated";
  if (value >= 0.1) return "risk-watch";
  return "risk-low";
}

function labelForPrediction(value: number) {
  return value === 1 ? "Fraud" : "Not fraud";
}

function metricValue(metrics: MetricsResponse, model: ModelKey, metric: string) {
  return metrics.models[model].holdout.metrics[metric]?.value;
}

function cvValue(metrics: MetricsResponse, model: ModelKey, metric: string) {
  return metrics.models[model].cv.metrics[metric]?.mean;
}

function cvStdValue(metrics: MetricsResponse, model: ModelKey, metric: string) {
  return metrics.models[model].cv.metrics[metric]?.std;
}

function confusionCount(metrics: MetricsResponse, model: ModelKey, metric: string) {
  return metrics.models[model].holdout.confusion_matrix[metric]?.count ?? 0;
}

function App() {
  const [activeView, setActiveView] = useState<ViewKey>("overview");
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [basicFeatures, setBasicFeatures] = useState<FeaturesResponse | null>(null);
  const [transactions, setTransactions] = useState<TransactionsResponse | null>(null);
  const [selectedTransaction, setSelectedTransaction] = useState<TransactionRow | null>(null);
  const [query, setQuery] = useState<TransactionQuery>({
    offset: 0,
    limit: 10,
    threshold: 0.5,
    sort: "disagreement",
  });
  const [overviewModel, setOverviewModel] = useState<ModelKey>("basic");
  const [thresholdModel, setThresholdModel] = useState<ModelKey>("full");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Initial data load keeps the app shell responsive while API data arrives.
    Promise.all([getMetrics(), getBasicFeatures()])
      .then(([metricsResult, featuresResult]) => {
        setMetrics(metricsResult);
        setBasicFeatures(featuresResult);
        setError(null);
      })
      .catch((caughtError: Error) => setError(caughtError.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    getTransactions(query)
      .then((result) => {
        setTransactions(result);
        setSelectedTransaction((current) => current ?? result.items[0] ?? null);
        setError(null);
      })
      .catch((caughtError: Error) => setError(caughtError.message));
  }, [query]);

  const appReady = metrics && basicFeatures && transactions;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <ShieldCheck size={24} />
          <div>
            <strong>Fraud Console</strong>
            <span>IEEE-CIS case study</span>
          </div>
        </div>

        <nav className="nav-list" aria-label="Primary">
          {views.map((view) => {
            const Icon = view.icon;
            return (
              <button
                key={view.key}
                className={activeView === view.key ? "nav-item active" : "nav-item"}
                onClick={() => setActiveView(view.key)}
              >
                <Icon size={17} />
                {view.label}
              </button>
            );
          })}
        </nav>

      </aside>

      <main className="workspace">
        <header className="topbar app-controls">
          {appReady && activeView === "overview" && (
            <ModelToggle model={overviewModel} onModelChange={setOverviewModel} />
          )}
          {appReady && activeView === "thresholds" && (
            <ModelToggle model={thresholdModel} onModelChange={setThresholdModel} />
          )}
        </header>

        {loading && <StatePanel title="Loading console" body="Reading model artefacts and holdout rows." />}
        {error && <StatePanel title="API connection issue" body={error} tone="danger" />}

        {appReady && activeView === "overview" && (
          <Overview
            metrics={metrics}
            model={overviewModel}
            onModelChange={setOverviewModel}
          />
        )}
        {appReady && activeView === "data" && <DataView metrics={metrics} />}
        {appReady && activeView === "transactions" && (
          <TransactionsView
            query={query}
            result={transactions}
            selected={selectedTransaction}
            onQueryChange={setQuery}
            onSelect={setSelectedTransaction}
          />
        )}
        {appReady && activeView === "thresholds" && (
          <ThresholdsView
            metrics={metrics}
            model={thresholdModel}
          />
        )}
        {appReady && activeView === "custom" && <CustomCheck features={basicFeatures} />}
      </main>
    </div>
  );
}

function StatePanel({ title, body, tone = "neutral" }: { title: string; body: string; tone?: "neutral" | "danger" }) {
  return (
    <section className={`state-panel ${tone}`}>
      <AlertTriangle size={20} />
      <div>
        <h2>{title}</h2>
        <p>{body}</p>
      </div>
    </section>
  );
}

function Overview({
  metrics,
  model,
  onModelChange,
}: {
  metrics: MetricsResponse;
  model: ModelKey;
  onModelChange: (model: ModelKey) => void;
}) {
  const modelDetails = {
    basic: {
      title: "Basic model",
      subtitle: "A logistic regression model with 14 simple features.",
    },
    full: {
      title: "Full model",
      subtitle: "An extreme gradient boosted tree classifier with 391 features.",
    },
  };
  const performanceRows = [
    {
      metric: "ROC AUC",
      cv: cvValue(metrics, model, "roc_auc"),
      cvStd: cvStdValue(metrics, model, "roc_auc"),
      holdout: metricValue(metrics, model, "roc_auc"),
    },
    {
      metric: "Average precision",
      cv: cvValue(metrics, model, "average_precision"),
      cvStd: cvStdValue(metrics, model, "average_precision"),
      holdout: metricValue(metrics, model, "average_precision"),
    },
    {
      metric: "Precision",
      cv: cvValue(metrics, model, "precision"),
      cvStd: cvStdValue(metrics, model, "precision"),
      holdout: metricValue(metrics, model, "precision"),
    },
    {
      metric: "Recall",
      cv: cvValue(metrics, model, "recall"),
      cvStd: cvStdValue(metrics, model, "recall"),
      holdout: metricValue(metrics, model, "recall"),
    },
    {
      metric: "F1",
      cv: cvValue(metrics, model, "f1"),
      cvStd: cvStdValue(metrics, model, "f1"),
      holdout: metricValue(metrics, model, "f1"),
    },
  ];
  // Explains the model metrics in the operational fraud-review terms used by this console.
  const metricDescriptions = [
    {
      metric: "ROC AUC",
      body: "How well the model ranks fraudulent transactions above genuine ones across all possible alert thresholds.",
    },
    {
      metric: "Average precision",
      body: "How useful the high-risk queue is when fraud is rare; higher means fraud appears nearer the top of the review list.",
    },
    {
      metric: "Precision",
      body: "Of the transactions flagged as fraud at the current threshold, the share that really were fraud.",
    },
    {
      metric: "Recall",
      body: "Of all fraudulent transactions in the holdout set, the share the model actually caught.",
    },
    {
      metric: "F1",
      body: "A single balance between catching fraud and avoiding wasted reviews from false alerts.",
    },
  ];

  return (
    <div className="view-grid overview-view">
      <section className="panel chart-panel">
        <div className="section-heading">
          <div>
            <h2>{modelDetails[model].title}</h2>
            <p>{modelDetails[model].subtitle}</p>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={performanceRows}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="metric" />
            <YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} />
            <Tooltip
              formatter={(value, name, item) => {
                if (item.dataKey === "cv") {
                  return [`${formatPercent(Number(value), 1)} ± ${formatPercent(item.payload.cvStd, 1)}`, "CV mean ± std"];
                }
                return [formatPercent(Number(value), 1), name];
              }}
            />
            <Bar dataKey="cv" name="CV mean ± std" fill="#0f766e" radius={[4, 4, 0, 0]}>
              <ErrorBar dataKey="cvStd" width={4} stroke="#14532d" />
            </Bar>
            <Bar dataKey="holdout" name="Holdout" fill="#d97706" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <div className="metric-explainer-grid">
          {metricDescriptions.map((description) => (
            <article key={description.metric}>
              <h3>{description.metric}</h3>
              <p>{description.body}</p>
            </article>
          ))}
        </div>
      </section>

      <HoldoutFlow metrics={metrics} model={model} />

    </div>
  );
}

function ModelToggle({
  model,
  onModelChange,
}: {
  model: ModelKey;
  onModelChange: (model: ModelKey) => void;
}) {
  return (
    <div className="overview-control">
      <div className="segmented-control">
        <button className={model === "basic" ? "active" : ""} onClick={() => onModelChange("basic")}>Basic</button>
        <button className={model === "full" ? "active" : ""} onClick={() => onModelChange("full")}>Full</button>
      </div>
    </div>
  );
}

function HoldoutFlow({ metrics, model }: { metrics: MetricsResponse; model: ModelKey }) {
  const truePositives = confusionCount(metrics, model, "true_positives");
  const trueNegatives = confusionCount(metrics, model, "true_negatives");
  const falsePositives = confusionCount(metrics, model, "false_positives");
  const falseNegatives = confusionCount(metrics, model, "false_negatives");
  const predictedFraud = truePositives + falsePositives;
  const predictedNotFraud = trueNegatives + falseNegatives;
  const total = predictedFraud + predictedNotFraud;
  const sankeyData = {
    nodes: [
      { name: "Total holdout", count: total, tone: "neutral" },
      { name: "Predicted fraud", count: predictedFraud, tone: "fraud" },
      { name: "Predicted not fraud", count: predictedNotFraud, tone: "notFraud" },
      { name: "True", count: truePositives, tone: "true", detail: "Caught fraud" },
      { name: "False", count: falsePositives, tone: "false", detail: "False alert" },
      { name: "True", count: trueNegatives, tone: "true", detail: "Actual not fraud" },
      { name: "False", count: falseNegatives, tone: "false", detail: "Missed fraud" },
    ],
    links: [
      { source: 0, target: 1, value: predictedFraud, tone: "fraud" },
      { source: 0, target: 2, value: predictedNotFraud, tone: "notFraud" },
      { source: 1, target: 3, value: truePositives, tone: "true" },
      { source: 1, target: 4, value: falsePositives, tone: "false" },
      { source: 2, target: 5, value: trueNegatives, tone: "true" },
      { source: 2, target: 6, value: falseNegatives, tone: "false" },
    ],
  };

  return (
    <section className="panel flow-panel">
        <div className="section-heading">
          <div>
            <h2>Holdout decision flow</h2>
          </div>
        </div>

      <div className="sankey-wrap" role="img" aria-label="Holdout rows split from total to model predictions and true or false outcomes">
        <ResponsiveContainer width="100%" height={460}>
          <Sankey
            data={sankeyData}
            node={SankeyNode}
            link={SankeyLink}
            nodePadding={70}
            nodeWidth={12}
            iterations={48}
            linkCurvature={0.42}
            sort={false}
            verticalAlign="justify"
            margin={{ top: 46, right: 190, bottom: 46, left: 150 }}
          >
            <Tooltip
              formatter={(value) => formatNumber(Number(value))}
              labelFormatter={(label) => String(label)}
            />
          </Sankey>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function SankeyNode({ x, y, width, height, payload }: SankeyNodeProps) {
  const chartWidth = useChartWidth() ?? 0;
  const nodePayload = payload as typeof payload & { tone?: string; count?: number; detail?: string };
  const isRightEdge = nodePayload.depth >= 2;
  const labelX = isRightEdge ? Math.min(x + width + 42, chartWidth - 24) : x + width + 10;
  const labelAnchor = "start";
  const nodeTone = String(nodePayload.tone ?? "neutral");
  const count = Number(nodePayload.count ?? nodePayload.value ?? 0);
  const nodeCentreY = y + height / 2;
  const textY = isRightEdge ? nodeCentreY : y + Math.max(18, Math.min(height / 2, height - 8));

  return (
    <g className={`sankey-node sankey-node-${nodeTone}`}>
      <rect x={x} y={y} width={width} height={height} rx={5} />
      <text className="sankey-node-label" x={labelX} y={textY - 22} textAnchor={labelAnchor}>
        {payload.name}
      </text>
      <text className="sankey-node-value" x={labelX} y={textY - 3} textAnchor={labelAnchor}>
        {formatNumber(count)}
      </text>
      {nodePayload.detail && (
        <text className="sankey-node-detail" x={labelX} y={textY + 15} textAnchor={labelAnchor}>
          {nodePayload.detail}
        </text>
      )}
    </g>
  );
}

function SankeyLink({
  sourceX,
  targetX,
  sourceY,
  targetY,
  sourceControlX,
  targetControlX,
  linkWidth,
  payload,
}: SankeyLinkProps) {
  const linkPayload = payload as typeof payload & { tone?: string };
  const linkTone = String(linkPayload.tone ?? "neutral");
  const halfWidth = Math.max(1, linkWidth) / 2;
  const path = [
    `M${sourceX},${sourceY - halfWidth}`,
    `C${sourceControlX},${sourceY - halfWidth} ${targetControlX},${targetY - halfWidth} ${targetX},${targetY - halfWidth}`,
    `L${targetX},${targetY + halfWidth}`,
    `C${targetControlX},${targetY + halfWidth} ${sourceControlX},${sourceY + halfWidth} ${sourceX},${sourceY + halfWidth}`,
    "Z",
  ].join(" ");

  return (
    <path
      className={`sankey-link sankey-link-${linkTone}`}
      d={path}
    />
  );
}

function DataView({ metrics }: { metrics: MetricsResponse }) {
  const basicFeatureNotes = [
    {
      title: "TransactionAmt",
      body: "Transaction payment amount in USD. Some values have three decimal places, which may reflect foreign-currency conversion or exchange-rate effects.",
    },
    {
      title: "ProductCD",
      body: "Product code for the transaction. In this dataset, product may mean a service or transaction category rather than a literal basket item.",
    },
    {
      title: "card1-card6",
      body: "Masked payment card information such as card type, card category, issuing bank, country, or related card attributes.",
    },
    {
      title: "addr, dist, email",
      body: "Address, distance, and email-domain signals. addr1/addr2 are purchaser billing region/country; dist fields can relate billing, mailing, ZIP, IP, or phone-area distances. P_emaildomain is purchaser email; R_emaildomain is recipient email and can be missing.",
    },
  ];

  const fullFeatureNotes = [
    {
      title: "C1-C14",
      body: "Masked count features, such as counts of addresses, devices, IPs, emails, names, or other transaction entities.",
    },
    {
      title: "D1-D15",
      body: "Masked timedelta features, such as days since previous transaction or other historical timing relationships.",
    },
    {
      title: "M1-M9",
      body: "Masked match features, such as whether names, addresses, card details, and related transaction entities match.",
    },
    {
      title: "Vesta Vxxx",
      body: "Engineered numerical Vesta features, including ranking, counting, clustering, time-window, and entity-relation signals.",
    },
  ];

  const identityFeatureNotes = [
    {
      title: "id_01-id_38",
      body: "Masked identity and verification signals tied to the transaction when an identity row is available.",
    },
    {
      title: "DeviceType",
      body: "Device category for the transaction, such as desktop or mobile, with missing values retained as a model signal.",
    },
    {
      title: "DeviceInfo",
      body: "Device, browser, operating-system, or user-agent style values, including high-cardinality strings such as Windows, iOS Device, MacOS, and phone build names.",
    },
    {
      title: "has_identity",
      body: "Explicit flag showing whether the transaction joined to the identity table, so identity coverage is visible rather than hidden inside missing values.",
    },
  ];

  const identityFeatureCount = 41;

  return (
    <div className="view-grid">
      <section className="panel about-columns">
        <article>
          <h3>What the app demonstrates</h3>
          <p>
            The dashboard shows how fraud risk changes as richer transaction and identity signals are added. It compares
            a compact baseline model with a full model, then lets you inspect review thresholds, prediction confidence,
            and example transactions.
          </p>
        </article>
        <article>
          <h3>How the models were built</h3>
          <p>
            The notebook joins the source data, builds numeric and categorical preprocessing pipelines,
            compares a compact logistic-regression baseline with an XGBoost model, evaluates on a stratified holdout
            split, then refits the deployable models on all labelled training rows.
          </p>
        </article>
        <article>
          <h3>How to read the scores</h3>
          <p>
            Probabilities are risk scores, not guarantees. A higher threshold reduces review load but misses more
            fraud; a lower threshold catches more fraud at the cost of more false positives.
          </p>
        </article>
      </section>

      <section className="panel data-intro">
        <h2>Dataset</h2>
        <p>
          The app uses the{" "}
          <a href="https://www.kaggle.com/competitions/ieee-fraud-detection" target="_blank" rel="noreferrer">
            IEEE-CIS transaction table
          </a>
          . The modelling notebook keeps a stratified 20% holdout split untouched until final evaluation, which is used
          for demonstration in this dashboard.
        </p>
      </section>

      <section className="metrics-strip data-metrics">
        <MetricCard label="Training rows" value={formatNumber(metrics.dataset.train_rows)} icon={Database} />
        <MetricCard label="Holdout rows" value={formatNumber(metrics.dataset.holdout_rows)} icon={Layers3} />
        <MetricCard label="Holdout fraud rate" value={formatPercent(metrics.dataset.holdout_fraud_rate, 2)} icon={AlertTriangle} />
      </section>

      <section className="panel column-guide">
        <div className="section-heading">
          <div>
            <h2>{formatNumber(metrics.models.basic.features_count)} basic features</h2>
          </div>
        </div>
        <div className="column-guide-grid">
          {basicFeatureNotes.map((note) => (
            <article key={note.title}>
              <h3>{note.title}</h3>
              <p>{note.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="panel column-guide">
        <div className="section-heading">
          <div>
            <h2>{formatNumber(metrics.models.full.features_count)} full features</h2>
            <p>All {formatNumber(metrics.models.basic.features_count)} basic features, plus:</p>
          </div>
        </div>
        <div className="column-guide-grid">
          {fullFeatureNotes.map((note) => (
            <article key={note.title}>
              <h3>{note.title}</h3>
              <p>{note.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="panel column-guide identity-guide">
        <div className="section-heading">
          <div>
            <h2>{formatNumber(identityFeatureCount)} identity features</h2>
            <p>Joined from the identity table when available</p>
          </div>
        </div>
        <div className="column-guide-grid">
          {identityFeatureNotes.map((note) => (
            <article key={note.title}>
              <h3>{note.title}</h3>
              <p>{note.body}</p>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function MetricCard({ label, value, icon: Icon }: { label: string; value: string; icon: LucideIcon }) {
  return (
    <article className="metric-card">
      <Icon size={18} />
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function ModelCard({
  model,
  metrics,
  title,
  subtitle,
}: {
  model: ModelKey;
  metrics: MetricsResponse;
  title: string;
  subtitle: string;
}) {
  return (
    <article className="model-card">
      <div className="model-card-header">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        <span>{metrics.models[model].model_type}</span>
      </div>
      <div className="model-stats">
        <SmallStat label="CV ROC AUC" value={formatPercent(cvValue(metrics, model, "roc_auc"), 1)} />
        <SmallStat label="Holdout ROC AUC" value={formatPercent(metricValue(metrics, model, "roc_auc"), 1)} />
        <SmallStat label="Holdout recall" value={formatPercent(metricValue(metrics, model, "recall"), 1)} />
      </div>
    </article>
  );
}

function SmallStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="small-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function TransactionsView({
  query,
  result,
  selected,
  onQueryChange,
  onSelect,
}: {
  query: TransactionQuery;
  result: TransactionsResponse;
  selected: TransactionRow | null;
  onQueryChange: (query: TransactionQuery) => void;
  onSelect: (row: TransactionRow) => void;
}) {
  const pageSize = query.limit ?? 10;
  const threshold = query.threshold ?? 0.5;
  const page = Math.floor((query.offset ?? 0) / pageSize) + 1;
  const totalPages = Math.max(1, Math.ceil(result.total / pageSize));

  function updateQuery(next: Partial<TransactionQuery>) {
    // Reset pagination whenever filters change so the table cannot land past the final page.
    onQueryChange({ ...query, ...next, offset: next.offset ?? 0 });
  }

  function changePage(nextPage: number) {
    // Convert the selected 1-based page number into the API's zero-based row offset.
    updateQuery({ offset: (nextPage - 1) * pageSize });
  }

  return (
    <div className="transactions-layout">
      <section className="panel table-panel">
        <div className="section-heading transaction-heading">
          <div>
            <h2>Holdout transaction explorer</h2>
            <div className="filter-row">
              <select value={query.sort ?? ""} onChange={(event) => updateQuery({ sort: event.target.value as TransactionQuery["sort"] || undefined })}>
                <option value="">Original order</option>
                <option value="disagreement">Largest disagreement</option>
                <option value="basic_probability">Basic risk</option>
                <option value="full_probability">Full risk</option>
              </select>
              <select value={query.actual ?? ""} onChange={(event) => updateQuery({ actual: event.target.value as TransactionQuery["actual"] || undefined })}>
                <option value="">All actual labels</option>
                <option value="fraud">Actual fraud</option>
                <option value="not_fraud">Actual not fraud</option>
              </select>
              <select value={threshold} onChange={(event) => updateQuery({ threshold: Number(event.target.value) })}>
                {TRANSACTION_THRESHOLDS.map((value) => (
                  <option key={value} value={value}>
                    {formatPercent(value, 0)} threshold
                  </option>
                ))}
              </select>
              <select value={query.basic_prediction ?? ""} onChange={(event) => updateQuery({ basic_prediction: event.target.value as TransactionQuery["basic_prediction"] || undefined })}>
                <option value="">All basic predictions</option>
                <option value="fraud">Basic predicts fraud</option>
                <option value="not_fraud">Basic predicts not fraud</option>
              </select>
              <select value={query.full_prediction ?? ""} onChange={(event) => updateQuery({ full_prediction: event.target.value as TransactionQuery["full_prediction"] || undefined })}>
                <option value="">All full predictions</option>
                <option value="fraud">Full predicts fraud</option>
                <option value="not_fraud">Full predicts not fraud</option>
              </select>
            </div>
          </div>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Transaction</th>
                <th>Actual</th>
                <th>Basic</th>
                <th>Full</th>
                <th>Gap</th>
              </tr>
            </thead>
            <tbody>
              {result.items.map((row) => {
                const gap = Math.abs(Number(row.full_fraud_probability) - Number(row.basic_fraud_probability));
                return (
                  <tr
                    key={row.TransactionID}
                    className={selected?.TransactionID === row.TransactionID ? "selected" : ""}
                    onClick={() => onSelect(row)}
                  >
                    <td>{row.TransactionID}</td>
                    <td><StatusPill value={labelForPrediction(Number(row.isFraud))} active={Number(row.isFraud) === 1} /></td>
                    <td><ProbabilityPill probability={Number(row.basic_fraud_probability)} threshold={threshold} /></td>
                    <td><ProbabilityPill probability={Number(row.full_fraud_probability)} threshold={threshold} /></td>
                    <td>{probabilityPercent(gap).toFixed(1)} pts</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="pagination">
          <span>{formatNumber(result.total)} rows · page {page} of {totalPages}</span>
          <div className="pagination-controls">
            <button disabled={page <= 1} onClick={() => changePage(page - 1)}>Previous</button>
            <select aria-label="Page" value={page} onChange={(event) => changePage(Number(event.target.value))}>
              {Array.from({ length: totalPages }, (_, index) => index + 1).map((pageNumber) => (
                <option key={pageNumber} value={pageNumber}>
                  {pageNumber}
                </option>
              ))}
            </select>
            <button disabled={page >= totalPages} onClick={() => changePage(page + 1)}>Next</button>
          </div>
        </div>

        <TransactionBreakdownStats result={result} />
      </section>

      <TransactionDetail selected={selected} threshold={threshold} />
    </div>
  );
}

function TransactionBreakdownStats({ result }: { result: TransactionsResponse }) {
  const columns = [
    {
      label: "Actual labels",
      fraud: result.breakdown.actual.fraud,
      notFraud: result.breakdown.actual.not_fraud,
    },
    {
      label: "Basic predictions",
      fraud: result.breakdown.basic_prediction.fraud,
      notFraud: result.breakdown.basic_prediction.not_fraud,
    },
    {
      label: "Full predictions",
      fraud: result.breakdown.full_prediction.fraud,
      notFraud: result.breakdown.full_prediction.not_fraud,
    },
  ];

  return (
    <section className="transaction-breakdown-card">
      <table className="breakdown-table">
        <thead>
          <tr>
            <th>Breakdown</th>
            {columns.map((column) => (
              <th key={column.label}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row"><BreakdownLabel label="Fraud" tone="danger" /></th>
            {columns.map((column) => (
              <td key={column.label}><BreakdownValue count={column.fraud} total={result.total} /></td>
            ))}
          </tr>
          <tr>
            <th scope="row"><BreakdownLabel label="Not fraud" tone="safe" /></th>
            {columns.map((column) => (
              <td key={column.label}><BreakdownValue count={column.notFraud} total={result.total} /></td>
            ))}
          </tr>
        </tbody>
      </table>
    </section>
  );
}

function BreakdownLabel({ label, tone }: { label: string; tone: "danger" | "safe" }) {
  return <span className={`breakdown-label ${tone}`}>{label}</span>;
}

function BreakdownValue({
  count,
  total,
}: {
  count: number;
  total: number;
}) {
  const share = total > 0 ? count / total : 0;

  return (
    <span className="breakdown-value">
      <strong>{formatNumber(count)}</strong>
      <em>{formatPercent(share, 1)}</em>
    </span>
  );
}

function TransactionDetail({ selected, threshold }: { selected: TransactionRow | null; threshold: number }) {
  if (!selected) {
    return <section className="panel detail-panel">Select a transaction to inspect model behaviour.</section>;
  }

  const basicProbability = Number(selected.basic_fraud_probability);
  const fullProbability = Number(selected.full_fraud_probability);
  const basicPrediction = basicProbability >= threshold ? 1 : 0;
  const fullPrediction = fullProbability >= threshold ? 1 : 0;
  const basicCorrect = basicPrediction === Number(selected.isFraud);
  const fullCorrect = fullPrediction === Number(selected.isFraud);
  const topFields = ["TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"];

  return (
    <section className="panel detail-panel">
      <div className="detail-header">
        <div>
          <h2>Transaction {selected.TransactionID}</h2>
          <p>Relative time {formatNumber(Number(selected.TransactionDT))}</p>
        </div>
        <StatusPill value={labelForPrediction(Number(selected.isFraud))} active={Number(selected.isFraud) === 1} />
      </div>

      <ProbabilityScale label="Basic model" probability={basicProbability} prediction={basicPrediction} correct={basicCorrect} threshold={threshold} />
      <ProbabilityScale label="Full model" probability={fullProbability} prediction={fullPrediction} correct={fullCorrect} threshold={threshold} />

      <div className="field-grid">
        {topFields.map((field) => (
          <SmallStat key={field} label={field} value={String(selected[field] ?? "missing")} />
        ))}
      </div>
    </section>
  );
}

function StatusPill({ value, active }: { value: string; active: boolean }) {
  return <span className={active ? "status-pill active" : "status-pill"}>{value}</span>;
}

function ProbabilityPill({ probability, threshold }: { probability: number; threshold: number }) {
  const percent = probabilityPercent(probability);
  const band = probability < threshold ? "low" : "high";

  return <span className={`probability-pill ${band}`}>{percent.toFixed(1)}%</span>;
}

function ProbabilityScale({
  label,
  probability,
  prediction,
  correct,
  threshold = 0.5,
}: {
  label: string;
  probability: number;
  prediction: number;
  correct: boolean;
  threshold?: number;
}) {
  return (
    <div className="probability-block">
      <div className="probability-label">
        <span>{label}</span>
        <strong>{probabilityPercent(probability).toFixed(1)}%</strong>
      </div>
      <div className="risk-track" aria-label={`${label} fraud probability`}>
        <div className={`risk-fill ${classForRisk(probability)}`} style={{ width: `${probabilityPercent(probability)}%` }} />
        <span className="threshold-marker" style={{ left: `${probabilityPercent(threshold)}%` }} />
      </div>
      <div className="probability-foot">
        <span>Prediction: {labelForPrediction(prediction)}</span>
        <span>{correct ? "Correct" : "Missed"}</span>
      </div>
    </div>
  );
}

function ThresholdsView({
  metrics,
  model,
}: {
  metrics: MetricsResponse;
  model: ModelKey;
}) {
  const rows = metrics.models[model].holdout.thresholds;

  return (
    <div className="view-grid">
      <section className="panel">
        <div className="section-heading">
          <div>
            <h2>Threshold trade-offs</h2>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="threshold" tickFormatter={(value) => `${Number(value) * 100}%`} />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="alerts" name="Alerts" stroke="#0f766e" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="caught_fraud" name="Caught fraud" stroke="#be123c" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="missed_fraud" name="Missed fraud" stroke="#7c3aed" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="false_positives" name="False positives" stroke="#d97706" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </section>

      <section className="panel table-panel">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Threshold</th>
                <th>Alerts</th>
                <th>Caught fraud</th>
                <th>Missed fraud</th>
                <th>False positives</th>
                <th>Precision</th>
                <th>Recall</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row: ThresholdRow) => (
                <tr key={row.threshold}>
                  <td>{formatPercent(row.threshold, 0)}</td>
                  <td>{formatNumber(row.alerts)}</td>
                  <td>{formatNumber(row.caught_fraud)}</td>
                  <td>{formatNumber(row.missed_fraud)}</td>
                  <td>{formatNumber(row.false_positives)}</td>
                  <td>{formatPercent(row.precision, 1)}</td>
                  <td>{formatPercent(row.recall, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function CustomCheck({ features }: { features: FeaturesResponse }) {
  const defaults = useMemo(() => {
    const entries = features.features.map((feature) => {
      if (feature.type === "numeric") return [feature.name, feature.median ?? 0];
      return [feature.name, feature.sample_values?.[0] ?? "missing"];
    });
    return Object.fromEntries(entries) as Record<string, string | number | null>;
  }, [features]);

  const [formState, setFormState] = useState<Record<string, string | number | null>>(defaults);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function updateField(feature: FeatureMeta, value: string) {
    // Numeric values stay as strings until submission so users can edit freely.
    setFormState((current) => ({ ...current, [feature.name]: value }));
  }

  function pickMedianWeightedNumber(feature: FeatureMeta) {
    const median = feature.median ?? 0;
    const minimum = feature.minimum ?? median;
    const maximum = feature.maximum ?? median;
    const lowerDistance = Math.max(0, median - minimum);
    const upperDistance = Math.max(0, maximum - median);
    const direction = Math.random() < 0.5 ? -1 : 1;
    const availableDistance = direction < 0 ? lowerDistance : upperDistance;
    const spread = availableDistance * 0.5;
    const rawValue = median + direction * Math.random() ** 2 * spread;
    const boundedValue = Math.min(maximum, Math.max(minimum, rawValue));

    return Number.isFinite(boundedValue) ? Number(boundedValue.toFixed(3)) : median;
  }

  function makeRandomInputs() {
    // Uses feature metadata to create a plausible row without requiring users to know the dataset.
    return Object.fromEntries(
      features.features.map((feature) => {
        if (feature.type === "numeric") {
          return [feature.name, pickMedianWeightedNumber(feature)];
        }

        const values = feature.sample_values ?? ["missing"];
        return [feature.name, values[Math.floor(Math.random() * values.length)] ?? "missing"];
      }),
    ) as Record<string, string | number | null>;
  }

  async function runPrediction(payload: Record<string, string | number | null>) {
    setSubmitting(true);
    setError(null);

    try {
      const result = await predictBasic(payload);
      setPrediction(result);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "Prediction failed.");
    } finally {
      setSubmitting(false);
    }
  }

  async function randomiseAndRun() {
    const nextValues = makeRandomInputs();
    setFormState(nextValues);
    setPrediction(null);
    await runPrediction(nextValues);
  }

  async function submitForm(event: FormEvent) {
    event.preventDefault();
    await runPrediction(formState);
  }

  return (
    <div className="custom-layout">
      <section className="panel">
        <div className="section-heading">
          <div>
            <h2>Custom check against basic model</h2>
            <p>Enter some values and run a live prediction through the basic pipeline.</p>
          </div>
        </div>

        <form className="custom-form" onSubmit={submitForm}>
          {features.features.map((feature) => (
            <label key={feature.name}>
              <span>{feature.name}</span>
              {feature.type === "numeric" ? (
                <input
                  type="number"
                  step="any"
                  value={String(formState[feature.name] ?? "")}
                  onChange={(event) => updateField(feature, event.target.value)}
                />
              ) : (
                <select
                  value={String(formState[feature.name] ?? "")}
                  onChange={(event) => updateField(feature, event.target.value)}
                >
                  {(feature.sample_values ?? ["missing"]).map((value) => (
                    <option key={String(value)} value={String(value)}>{String(value)}</option>
                  ))}
                </select>
              )}
            </label>
          ))}

          <div className="form-actions">
            <button className="secondary-button" type="button" onClick={randomiseAndRun} disabled={submitting}>
              <Shuffle size={16} />
              {submitting ? "Checking" : "Randomise and run"}
            </button>
            <button className="primary-button" type="submit" disabled={submitting}>
              <BarChart3 size={17} />
              {submitting ? "Checking" : "Run basic check"}
            </button>
          </div>
        </form>
      </section>

      <section className="panel result-panel">
        {prediction ? (
          <>
            <div className="result-score">
              <span>{prediction.risk_band}</span>
              <strong>{probabilityPercent(prediction.fraud_probability).toFixed(1)}%</strong>
              <p>Fraud probability at a {formatPercent(prediction.threshold, 0)} decision threshold.</p>
            </div>
            <ProbabilityScale
              label="Custom basic model"
              probability={prediction.fraud_probability}
              prediction={prediction.prediction}
              correct
            />
          </>
        ) : (
          <div className="empty-result">
            <Gauge size={34} />
            <h2>No custom score yet</h2>
            <p>Defaults use medians for numeric fields and common categories for categorical fields.</p>
          </div>
        )}
        {error && <p className="form-error">{error}</p>}
      </section>
    </div>
  );
}

export default App;
