const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export type ModelKey = "basic" | "full";
export type ViewKey = "overview" | "data" | "transactions" | "thresholds" | "custom";

export type MetricValue = {
  value?: number;
  mean?: number;
  std?: number;
  count?: number;
  percentage?: number;
};

export type ThresholdRow = {
  threshold: number;
  alerts: number;
  alert_rate: number;
  caught_fraud: number;
  missed_fraud: number;
  false_positives: number;
  precision: number;
  reviews_per_caught_fraud: number;
  recall: number;
  f1: number;
};

export type ModelMetrics = {
  model_type: string;
  features_count: number;
  cv: {
    folds: number;
    metrics: Record<string, MetricValue>;
  };
  holdout: {
    threshold: number;
    metrics: Record<string, MetricValue>;
    confusion_matrix: Record<string, MetricValue>;
    thresholds: ThresholdRow[];
  };
};

export type MetricsResponse = {
  dataset: {
    name: string;
    target: string;
    train_rows: number;
    holdout_rows: number;
    test_size: number;
    split_strategy: string;
    random_state: number;
    train_fraud_rate: number;
    holdout_fraud_rate: number;
  };
  models: Record<ModelKey, ModelMetrics>;
};

export type FeatureMeta = {
  name: string;
  dtype: string;
  type: "numeric" | "categorical";
  required: boolean;
  missing_percentage: number;
  unique_values: number;
  minimum?: number;
  maximum?: number;
  mean?: number;
  median?: number;
  sample_values?: Array<string | number>;
};

export type FeaturesResponse = {
  model_name: string;
  target: string;
  features_count: number;
  numeric_features: string[];
  categorical_features: string[];
  features: FeatureMeta[];
};

export type TransactionRow = {
  TransactionID: number;
  TransactionDT: number;
  isFraud: number;
  basic_prediction: number;
  basic_fraud_probability: number;
  full_prediction: number;
  full_fraud_probability: number;
  [key: string]: string | number | null;
};

export type PredictionFilter = "fraud" | "not_fraud";

export type TransactionBreakdown = {
  actual: Record<PredictionFilter, number>;
  basic_prediction: Record<PredictionFilter, number>;
  full_prediction: Record<PredictionFilter, number>;
};

export type TransactionsResponse = {
  offset: number;
  limit: number;
  total: number;
  breakdown: TransactionBreakdown;
  items: TransactionRow[];
};

export type TransactionQuery = {
  offset?: number;
  limit?: number;
  threshold?: number;
  actual?: PredictionFilter;
  basic_prediction?: PredictionFilter;
  full_prediction?: PredictionFilter;
  sort?: "basic_probability" | "full_probability" | "disagreement";
};

export type PredictionResponse = {
  fraud_probability: number;
  prediction: number;
  risk_band: "Low" | "Watch" | "Elevated" | "High";
  threshold: number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // Centralises API error handling so views can focus on display state.
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
    ...init,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function getMetrics() {
  return request<MetricsResponse>("/api/metrics");
}

export function getBasicFeatures() {
  return request<FeaturesResponse>("/api/features/basic");
}

export function getTransactions(query: TransactionQuery) {
  const params = new URLSearchParams();
  const allowedEntries = {
    offset: query.offset,
    limit: query.limit,
    threshold: query.threshold,
    actual: query.actual,
    basic_prediction: query.basic_prediction,
    full_prediction: query.full_prediction,
    sort: query.sort,
  };

  Object.entries(allowedEntries).forEach(([key, value]) => {
    // Blank select options mean "no filter"; omit them so FastAPI sees its defaults.
    if (value !== undefined && String(value) !== "") {
      params.set(key, String(value));
    }
  });

  return request<TransactionsResponse>(`/api/transactions?${params.toString()}`);
}

export function predictBasic(payload: Record<string, string | number | null>) {
  return request<PredictionResponse>("/api/predict/basic", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
