import { useEffect, useState, type FormEvent, type ReactNode } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

type Reading = {
  product_type: 'L' | 'M' | 'H'
  air_temperature: number
  process_temperature: number
  rotational_speed: number
  torque: number
  tool_wear: number
}

type ModelMetrics = {
  model: string
  accuracy: number
  precision: number
  recall: number
  f1: number
  roc_auc: number
  confusion_matrix: {
    true_negative: number
    false_positive: number
    false_negative: number
    true_positive: number
  }
}

type FeatureImportance = {
  feature: string
  importance: number
}

type Overview = {
  dataset: {
    name: string
    source: string
    rows: number
    failure_cases: number
    failure_rate: number
    training_rows: number
    test_rows: number
    split: string
  }
  selected_model: string
  cross_validation_f1: Record<string, number>
  classification_metrics: ModelMetrics[]
  anomaly_metrics: {
    precision: number
    recall: number
    f1: number
    flagged: number
    test_rows: number
    contamination_assumption: number
  }
  feature_importance: FeatureImportance[]
}

type Prediction = {
  model: string
  failure_score: number
  predicted_failure: boolean
  priority: 'Low' | 'Moderate' | 'High' | 'Critical'
  feature_importance: FeatureImportance[]
}

type PredictionExplanation = {
  model: string
  failure_score: number
  predicted_failure: boolean
  baseline_value: number
  contributions: {
    feature: string
    value: string | number
    shap_value: number
    direction: 'increases' | 'decreases' | 'no change'
  }[]
  interpretation: string
}

type AnomalyResult = {
  is_anomaly: boolean
  anomaly_score: number
  threshold_assumption: number
}

type ZscoreResult = {
  is_anomaly: boolean
  feature_zscores: Record<string, number>
  exceeded_features: string[]
  threshold: number
  max_absolute_zscore: number
  zero_variance_features?: string[]
}

type SimulationEvent = {
  simulated: boolean
  source: string
  asset_id: string
  simulation_mode?: 'normal' | 'abnormal'
  reading: Reading
  prediction: Prediction
  anomaly: AnomalyResult
  zscore: ZscoreResult
  recommendation?: {
    severity: 'routine' | 'warning' | 'urgent' | 'critical'
    actions: { code: string; title: string; detail: string }[]
    reasons: { code: string; message: string }[]
  }
  persisted?: boolean
  created_at?: string | null
  id?: number | null
}

type SimulationStatus = {
  simulator_running: boolean
  interval_seconds: number
  samples_published: number
  latest: SimulationEvent | null
}

type PersistedAssessment = {
  id: number
  created_at: string | null
  source: string
  asset_id: string
  kind: string
  reading: Record<string, unknown>
  prediction: Record<string, unknown> | null
  anomaly: Record<string, unknown> | null
  zscore: Record<string, unknown> | null
  recommendation: Record<string, unknown> | null
  rul: Record<string, unknown> | null
  simulated: boolean
  persisted: boolean
}

type AssessmentHistoryResponse = {
  database_connected: boolean
  database_target: string
  database_error: string | null
  items: PersistedAssessment[]
  total: number
  limit: number
  offset: number
}

type AssessmentAnalytics = {
  total_assessments: number
  predicted_failures: number
  normal_assessments: number
  anomaly_assessments: number
  urgent_critical: number
  monitored_assets: number
  simulated_assessments: number
  persisted_assessments: number
  activity_by_day: { date: string; count: number }[]
  failure_distribution: { predicted_failure: number; normal: number }
  anomaly_distribution: {
    any_anomaly: number
    isolation_forest: number
    zscore: number
    normal: number
  }
  severity_distribution: { routine: number; warning: number; urgent: number; critical: number }
  asset_comparison: { asset_id: string; count: number }[]
}

type AnalyticsResponse = {
  database_connected: boolean
  database_target: string
  database_error: string | null
  analytics: AssessmentAnalytics | null
}

type RulModelMetrics = {
  model: string
  rmse: number
  mae: number
  nasa_score: number
}

type RulOverview = {
  dataset: {
    name: string
    source: string
    source_url: string
    training_engines: number
    test_engines: number
    training_observations: number
    test_observations: number
    operational_settings: number
    sensor_measurements: number
    rul_cap: number
    validation: string
    test_evaluation: string
  }
  selected_model: string
  validation_metrics: RulModelMetrics[]
  test_metrics: RulModelMetrics[]
  methodology: {
    training_target: string
    test_target: string
    test_evaluation: string
    nasa_score: string
  }
}

type RulPrediction = {
  predicted_rul: number
  model: string
  interpretation: string
  units: 'cycles'
  capped_upper_bound: number
}

type RulReading = {
  operational_setting_1: number
  operational_setting_2: number
  operational_setting_3: number
  sensor_1: number
  sensor_2: number
  sensor_3: number
  sensor_4: number
  sensor_5: number
  sensor_6: number
  sensor_7: number
  sensor_8: number
  sensor_9: number
  sensor_10: number
  sensor_11: number
  sensor_12: number
  sensor_13: number
  sensor_14: number
  sensor_15: number
  sensor_16: number
  sensor_17: number
  sensor_18: number
  sensor_19: number
  sensor_20: number
  sensor_21: number
}

type Page = 'overview' | 'predict' | 'evaluation' | 'anomaly' | 'rul' | 'simulation' | 'history' | 'about'

const initialReading: Reading = {
  product_type: 'M',
  air_temperature: 300,
  process_temperature: 310,
  rotational_speed: 1500,
  torque: 40,
  tool_wear: 100,
}

const initialRulReading: RulReading = {
  operational_setting_1: -0.0007,
  operational_setting_2: -0.0004,
  operational_setting_3: 100,
  sensor_1: 518.67,
  sensor_2: 641.82,
  sensor_3: 1589.70,
  sensor_4: 1400.60,
  sensor_5: 14.62,
  sensor_6: 21.61,
  sensor_7: 554.36,
  sensor_8: 2388.06,
  sensor_9: 9046.19,
  sensor_10: 1.30,
  sensor_11: 47.47,
  sensor_12: 521.66,
  sensor_13: 2388.02,
  sensor_14: 8138.62,
  sensor_15: 8.4195,
  sensor_16: 0.03,
  sensor_17: 392,
  sensor_18: 2388,
  sensor_19: 100,
  sensor_20: 39.06,
  sensor_21: 23.419,
}

const numericInputs: {
  key: Exclude<keyof Reading, 'product_type'>
  label: string
  unit: string
  min: number
  max: number
  step: number
}[] = [
  { key: 'air_temperature', label: 'Air temperature', unit: 'K', min: 250, max: 350, step: 0.1 },
  { key: 'process_temperature', label: 'Process temperature', unit: 'K', min: 250, max: 400, step: 0.1 },
  { key: 'rotational_speed', label: 'Rotational speed', unit: 'rpm', min: 100, max: 5000, step: 1 },
  { key: 'torque', label: 'Torque', unit: 'Nm', min: 0, max: 300, step: 0.1 },
  { key: 'tool_wear', label: 'Tool wear', unit: 'min', min: 0, max: 300, step: 1 },
]

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  if (init?.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
  })
  const payload = (await response.json()) as T & { detail?: string }
  if (!response.ok) {
    throw new Error(payload.detail ?? `Request failed (${response.status})`)
  }
  return payload
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`
}

function Icon({ children }: { children: ReactNode }) {
  return <span className="nav-icon" aria-hidden="true">{children}</span>
}

function App() {
  const [page, setPage] = useState<Page>('overview')
  const [overview, setOverview] = useState<Overview | null>(null)
  const [rulOverview, setRulOverview] = useState<RulOverview | null>(null)
  const [rulOverviewLoading, setRulOverviewLoading] = useState(true)
  const [rulOverviewError, setRulOverviewError] = useState<string | null>(null)
  const [reading, setReading] = useState<Reading>(initialReading)
  const [rulReading, setRulReading] = useState<RulReading>(initialRulReading)
  const [rulPrediction, setRulPrediction] = useState<RulPrediction | null>(null)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [predictionReading, setPredictionReading] = useState<Reading | null>(null)
  const [explanation, setExplanation] = useState<PredictionExplanation | null>(null)
  const [explaining, setExplaining] = useState(false)
  const [explanationError, setExplanationError] = useState<string | null>(null)
  const [anomaly, setAnomaly] = useState<AnomalyResult | null>(null)
  const [zscore, setZscore] = useState<ZscoreResult | null>(null)
  const [simulationStatus, setSimulationStatus] = useState<SimulationStatus | null>(null)
  const [simulationHistory, setSimulationHistory] = useState<SimulationEvent[]>([])
  const [simulationPollingEnabled, setSimulationPollingEnabled] = useState(false)
  const [assessmentHistory, setAssessmentHistory] = useState<PersistedAssessment[]>([])
  const [assessmentHistoryMeta, setAssessmentHistoryMeta] = useState<AssessmentHistoryResponse | null>(null)
  const [historyAssetFilter, setHistoryAssetFilter] = useState('')
  const [historyKindFilter, setHistoryKindFilter] = useState('')
  const [historyStartFilter, setHistoryStartFilter] = useState('')
  const [historyEndFilter, setHistoryEndFilter] = useState('')
  const [assessmentAnalytics, setAssessmentAnalytics] = useState<AssessmentAnalytics | null>(null)
  const [analyticsAvailable, setAnalyticsAvailable] = useState(true)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    requestJson<Overview>('/api/overview')
      .then(setOverview)
      .catch((cause: unknown) => setError(cause instanceof Error ? cause.message : 'Could not load benchmark data.'))
      .finally(() => setLoading(false))
    requestJson<RulOverview>('/api/rul/overview')
      .then(setRulOverview)
      .catch((cause: unknown) => {
        const message = cause instanceof Error ? cause.message : 'Could not load NASA RUL data.'
        setRulOverviewError(message)
        setError(message)
      })
      .finally(() => setRulOverviewLoading(false))
    requestJson<AnalyticsResponse>('/api/analytics')
      .then((response) => {
        setAnalyticsAvailable(response.database_connected)
        setAssessmentAnalytics(response.analytics)
      })
      .catch((cause: unknown) => {
        setAnalyticsAvailable(false)
        setError(cause instanceof Error ? cause.message : 'Could not load persisted analytics.')
      })
  }, [])

  useEffect(() => {
    if (!simulationPollingEnabled) {
      return
    }
    const refreshSimulation = () => {
      Promise.all([
        requestJson<SimulationStatus>('/api/stream/status'),
        requestJson<{ items: SimulationEvent[] }>('/api/simulation/history?limit=20'),
      ]).then(([status, history]) => {
        setSimulationStatus(status)
        setSimulationHistory(history.items)
      }).catch(() => {
        // The simulation panel retains its last known state.
      })
    }
    refreshSimulation()
    const intervalId = window.setInterval(refreshSimulation, 2000)
    return () => {
      window.clearInterval(intervalId)
    }
  }, [simulationPollingEnabled])

  async function loadAssessmentHistory(offset = 0) {
    try {
      const params = new URLSearchParams({ limit: '50', offset: String(offset) })
      if (historyAssetFilter) params.set('asset_id', historyAssetFilter)
      if (historyKindFilter) params.set('kind', historyKindFilter)
      if (historyStartFilter) params.set('start', historyStartFilter)
      if (historyEndFilter) params.set('end', historyEndFilter)
      const response = await requestJson<AssessmentHistoryResponse>(
        `/api/assessments?${params.toString()}`,
      )
      setAssessmentHistoryMeta(response)
      setAssessmentHistory(response.items)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not load assessment history.')
    }
  }

  async function submitPrediction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const submittedReading = { ...reading }
    setSubmitting(true)
    setError(null)
    try {
      const result = await requestJson<Prediction>('/api/predict', {
        method: 'POST',
        body: JSON.stringify(submittedReading),
      })
      setPrediction(result)
      setPredictionReading(submittedReading)
      setExplanation(null)
      setExplanationError(null)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Prediction request failed.')
    } finally {
      setSubmitting(false)
    }
  }

  async function explainPrediction() {
    if (!predictionReading) {
      setExplanationError('Run a failure prediction before requesting an explanation.')
      return
    }

    setExplaining(true)
    setExplanationError(null)
    setExplanation(null)
    try {
      setExplanation(await requestJson<PredictionExplanation>('/api/explain', {
        method: 'POST',
        body: JSON.stringify(predictionReading),
      }))
    } catch (cause) {
      setExplanationError(cause instanceof Error ? cause.message : 'Explanation request failed.')
    } finally {
      setExplaining(false)
    }
  }

  async function submitAnomaly(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setSubmitting(true)
    setError(null)
    try {
      setAnomaly(await requestJson<AnomalyResult>('/api/anomaly', {
        method: 'POST',
        body: JSON.stringify(reading),
      }))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Anomaly request failed.')
    } finally {
      setSubmitting(false)
    }
  }

  async function submitZscore() {
    setSubmitting(true)
    setError(null)
    try {
      setZscore(await requestJson<ZscoreResult>('/api/zscore', {
        method: 'POST',
        body: JSON.stringify(reading),
      }))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Z-score request failed.')
    } finally {
      setSubmitting(false)
    }
  }

  function updateNumber(key: Exclude<keyof Reading, 'product_type'>, value: string) {
    setReading((current) => ({ ...current, [key]: Number(value) }))
  }

  const selectedMetrics = overview?.classification_metrics.find(
    (metric) => metric.model === overview.selected_model,
  )

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">IM</div>
          <div>
            <strong>IntelliMaint</strong>
            <span>Predictive intelligence</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <nav className="navigation" aria-label="Main navigation">
          <NavButton active={page === 'overview'} onClick={() => setPage('overview')} icon="◫">Overview</NavButton>
          <NavButton active={page === 'predict'} onClick={() => setPage('predict')} icon="⌁">Failure prediction</NavButton>
          <NavButton active={page === 'evaluation'} onClick={() => setPage('evaluation')} icon="▥">Model evaluation</NavButton>
          <NavButton active={page === 'anomaly'} onClick={() => setPage('anomaly')} icon="⌖">Anomaly detection</NavButton>
          <NavButton active={page === 'rul'} onClick={() => setPage('rul')} icon="◷">RUL estimation</NavButton>
          <NavButton active={page === 'simulation'} onClick={() => setPage('simulation')} icon="◉">Simulated monitoring</NavButton>
          <NavButton active={page === 'history'} onClick={() => {
            setPage('history')
            void loadAssessmentHistory()
          }} icon="▤">Assessment history</NavButton>
        </nav>

        <div className="sidebar-bottom">
          <div className="scope-card">
            <span className="scope-dot" />
            <div>
              <strong>Benchmark mode</strong>
              <span>AI4I 2020 dataset</span>
            </div>
          </div>
          <NavButton active={page === 'about'} onClick={() => setPage('about')} icon="?">Project notes</NavButton>
          <div className="sidebar-footer">Research prototype · v0.1</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="eyebrow">INDUSTRIAL ANALYTICS <span>/</span> BENCHMARK LAB</div>
            <h1>{pageTitle(page)}</h1>
          </div>
          <div className="topbar-right">
            <span className={`connection-dot ${overview ? 'connected' : ''}`} />
            <span>{overview ? 'API connected' : 'Connecting to API'}</span>
            <div className="avatar">AI</div>
          </div>
        </header>

        {error && (
          <div className="error-banner" role="alert">
            <strong>Could not complete the request.</strong> {error}
          </div>
        )}

        {loading ? (
          <div className="loading-panel"><span className="spinner" /> Loading dataset and training models…</div>
        ) : !overview ? (
          <div className="empty-panel">
            <h2>Backend not ready</h2>
            <p>Start the FastAPI server and confirm the AI4I CSV is available, then reload this page.</p>
            <code>cd backend; ..\venv\Scripts\python.exe -m uvicorn app.main:app --reload</code>
          </div>
        ) : (
          <div className="page-content">
            {page === 'overview' && (
              <OverviewPage
                overview={overview}
                selectedMetrics={selectedMetrics}
                onPredict={() => setPage('predict')}
                analytics={assessmentAnalytics}
                analyticsAvailable={analyticsAvailable}
              />
            )}
            {page === 'predict' && (
              <PredictPage
                reading={reading}
                prediction={prediction}
                explanation={explanation}
                explanationError={explanationError}
                explaining={explaining}
                hasPredictionReading={predictionReading !== null}
                submitting={submitting}
                onUpdateNumber={updateNumber}
                onUpdateType={(value) => setReading((current) => ({ ...current, product_type: value }))}
                onSubmit={submitPrediction}
                onExplain={explainPrediction}
                onReset={() => {
                  setPrediction(null)
                  setPredictionReading(null)
                  setExplanation(null)
                  setExplanationError(null)
                }}
              />
            )}
            {page === 'evaluation' && <EvaluationPage overview={overview} />}
            {page === 'anomaly' && (
              <AnomalyPage
                reading={reading}
                anomaly={anomaly}
                zscore={zscore}
                metrics={overview.anomaly_metrics}
                submitting={submitting}
                onUpdateNumber={updateNumber}
                onUpdateType={(value) => setReading((current) => ({ ...current, product_type: value }))}
                onSubmit={submitAnomaly}
                onSubmitZscore={submitZscore}
              />
            )}
            {page === 'rul' && (
              <RulPage
                overview={rulOverview}
                loading={rulOverviewLoading}
                error={rulOverviewError}
                reading={rulReading}
                prediction={rulPrediction}
                submitting={submitting}
                onUpdate={(key, value) => setRulReading((current) => ({ ...current, [key]: Number(value) }))}
                onSubmit={async (event) => {
                  event.preventDefault()
                  setSubmitting(true)
                  setError(null)
                  try {
                    setRulPrediction(await requestJson<RulPrediction>('/api/rul/predict', {
                      method: 'POST',
                      body: JSON.stringify(rulReading),
                    }))
                  } catch (cause) {
                    setError(cause instanceof Error ? cause.message : 'RUL request failed.')
                  } finally {
                    setSubmitting(false)
                  }
                }}
              />
            )}
            {page === 'simulation' && (
              <SimulationPage
                status={simulationStatus}
                history={simulationHistory}
                onStart={async () => {
                  setSubmitting(true)
                  setError(null)
                  try {
                    const status = await requestJson<SimulationStatus>('/api/stream/start', {
                      method: 'POST',
                      body: JSON.stringify({ interval_seconds: 2 }),
                    })
                    setSimulationStatus(status)
                    setSimulationPollingEnabled(true)
                  } catch (cause) {
                    setError(cause instanceof Error ? cause.message : 'Could not start simulation.')
                  } finally {
                    setSubmitting(false)
                  }
                }}
                onStop={async () => {
                  setSubmitting(true)
                  setError(null)
                  try {
                    const status = await requestJson<SimulationStatus>('/api/stream/stop', { method: 'POST' })
                    setSimulationStatus(status)
                    setSimulationPollingEnabled(false)
                  } catch (cause) {
                    setError(cause instanceof Error ? cause.message : 'Could not pause simulation.')
                  } finally {
                    setSubmitting(false)
                  }
                }}
                onGenerate={async (mode) => {
                  setSubmitting(true)
                  setError(null)
                  try {
                    await requestJson<SimulationEvent>(`/api/simulation/generate?mode=${mode}`, { method: 'POST' })
                    const [status, history] = await Promise.all([
                      requestJson<SimulationStatus>('/api/stream/status'),
                      requestJson<{ items: SimulationEvent[] }>('/api/simulation/history?limit=20'),
                    ])
                    setSimulationStatus(status)
                    setSimulationHistory(history.items)
                  } catch (cause) {
                    setError(cause instanceof Error ? cause.message : 'Could not generate simulation.')
                  } finally {
                    setSubmitting(false)
                  }
                }}
                onClear={async () => {
                  setSubmitting(true)
                  setError(null)
                  try {
                    await requestJson<{ reset: boolean }>('/api/simulation/reset', { method: 'POST' })
                    setSimulationHistory([])
                    setSimulationStatus((current) => current
                      ? { ...current, samples_published: 0, latest: null }
                      : current)
                  } catch (cause) {
                    setError(cause instanceof Error ? cause.message : 'Could not reset simulation.')
                  } finally {
                    setSubmitting(false)
                  }
                }}
                submitting={submitting}
              />
            )}
            {page === 'history' && (
              <AssessmentHistoryPage
                meta={assessmentHistoryMeta}
                items={assessmentHistory}
                onLoadMore={(offset) => void loadAssessmentHistory(offset)}
                assetFilter={historyAssetFilter}
                kindFilter={historyKindFilter}
                startFilter={historyStartFilter}
                endFilter={historyEndFilter}
                onAssetFilter={setHistoryAssetFilter}
                onKindFilter={setHistoryKindFilter}
                onStartFilter={setHistoryStartFilter}
                onEndFilter={setHistoryEndFilter}
                onApplyFilters={() => void loadAssessmentHistory(0)}
              />
            )}
            {page === 'about' && <AboutPage overview={overview} />}
          </div>
        )}
      </main>
    </div>
  )
}

function NavButton({
  active,
  onClick,
  icon,
  children,
}: {
  active: boolean
  onClick: () => void
  icon: string
  children: ReactNode
}) {
  return (
    <button className={`nav-button ${active ? 'active' : ''}`} onClick={onClick} type="button">
      <Icon>{icon}</Icon>
      <span>{children}</span>
      {active && <span className="nav-active-mark" />}
    </button>
  )
}

function pageTitle(page: Page) {
  const titles: Record<Page, string> = {
    overview: 'Health overview',
    predict: 'Failure risk assessment',
    evaluation: 'Model evaluation',
    anomaly: 'Anomaly detection',
    rul: 'NASA C-MAPSS RUL estimation',
    simulation: 'Simulated sensor monitoring',
    history: 'Persisted assessment history',
    about: 'Project notes',
  }
  return titles[page]
}

function OverviewPage({
  overview,
  selectedMetrics,
  onPredict,
  analytics,
  analyticsAvailable,
}: {
  overview: Overview
  selectedMetrics: ModelMetrics | undefined
  onPredict: () => void
  analytics: AssessmentAnalytics | null
  analyticsAvailable: boolean
}) {
  return (
    <>
      <section className="welcome-row">
        <div>
          <div className="eyebrow">MONITORING SNAPSHOT <span className="live-tag">BENCHMARK</span></div>
          <h2>Predictive maintenance intelligence</h2>
          <p>Explore model results and test sensor readings against the AI4I benchmark.</p>
        </div>
        <button className="primary-button" onClick={onPredict} type="button">
          <span>＋</span> Run a prediction
        </button>
      </section>

      {analyticsAvailable && (
        <section className="stat-grid persisted-stat-grid" aria-label="Persisted assessment KPIs">
          <StatCard
            label="Persisted assessments"
            value={(analytics?.total_assessments ?? 0).toLocaleString()}
            detail="PostgreSQL or configured database"
            icon="▤"
            accent="blue"
          />
          <StatCard
            label="Predicted failures"
            value={(analytics?.predicted_failures ?? 0).toLocaleString()}
            detail="Model-predicted, not confirmed failures"
            icon="⚠"
            accent="orange"
          />
          <StatCard
            label="Normal assessments"
            value={(analytics?.normal_assessments ?? 0).toLocaleString()}
            detail="No predicted-failure flag"
            icon="✓"
            accent="green"
          />
          <StatCard
            label="Anomalies detected"
            value={(analytics?.anomaly_assessments ?? 0).toLocaleString()}
            detail="Isolation Forest or Z-score"
            icon="⌖"
            accent="blue"
          />
          <StatCard
            label="Urgent / critical"
            value={(analytics?.urgent_critical ?? 0).toLocaleString()}
            detail="Highest maintenance severities"
            icon="!"
            accent="orange"
          />
          <StatCard
            label="Monitored assets"
            value={(analytics?.monitored_assets ?? 0).toLocaleString()}
            detail={`${(analytics?.simulated_assessments ?? 0).toLocaleString()} simulated persisted records`}
            icon="◉"
            accent="green"
          />
        </section>
      )}

      {analyticsAvailable && (
        <section className="content-grid analytics-grid" aria-label="Persisted assessment analytics">
          <div className="panel activity-panel">
            <PanelHeading title="Assessment activity" subtitle="Persisted assessments by day" />
            {analytics?.activity_by_day.length ? (
              <div className="activity-chart">
                {analytics.activity_by_day.map((item) => {
                  const maxCount = Math.max(...analytics.activity_by_day.map((entry) => entry.count), 1)
                  return (
                    <div className="activity-bar" key={item.date} title={`${item.date}: ${item.count}`}>
                      <div style={{ height: `${(item.count / maxCount) * 100}%` }} />
                      <span>{item.date.slice(5)}</span>
                    </div>
                  )
                })}
              </div>
            ) : (
              <p className="form-note">No persisted activity for the available records.</p>
            )}
          </div>
          <div className="panel distribution-panel">
            <PanelHeading title="Assessment distribution" subtitle="Predictions, anomalies and severity" />
            <DistributionBar
              label="Predicted failure"
              value={analytics?.failure_distribution.predicted_failure ?? 0}
              maxValue={Math.max(
                Number.isFinite(analytics?.failure_distribution.predicted_failure) ? Math.max(0, analytics?.failure_distribution.predicted_failure ?? 0) : 0,
                Number.isFinite(analytics?.failure_distribution.normal) ? Math.max(0, analytics?.failure_distribution.normal ?? 0) : 0,
              )}
              tone="orange"
            />
            <DistributionBar
              label="Normal prediction"
              value={analytics?.failure_distribution.normal ?? 0}
              maxValue={Math.max(
                Number.isFinite(analytics?.failure_distribution.predicted_failure) ? Math.max(0, analytics?.failure_distribution.predicted_failure ?? 0) : 0,
                Number.isFinite(analytics?.failure_distribution.normal) ? Math.max(0, analytics?.failure_distribution.normal ?? 0) : 0,
              )}
              tone="green"
            />
            <DistributionBar
              label="Anomaly"
              value={analytics?.anomaly_distribution.any_anomaly ?? 0}
              maxValue={Math.max(
                analytics?.anomaly_distribution.any_anomaly ?? 0,
                analytics?.anomaly_distribution.isolation_forest ?? 0,
                analytics?.anomaly_distribution.zscore ?? 0,
              )}
              tone="blue"
            />
            <DistributionBar
              label="Isolation Forest"
              value={analytics?.anomaly_distribution.isolation_forest ?? 0}
              maxValue={Math.max(
                analytics?.anomaly_distribution.any_anomaly ?? 0,
                analytics?.anomaly_distribution.isolation_forest ?? 0,
                analytics?.anomaly_distribution.zscore ?? 0,
              )}
              tone="blue"
            />
            <DistributionBar
              label="Z-score"
              value={analytics?.anomaly_distribution.zscore ?? 0}
              maxValue={Math.max(
                analytics?.anomaly_distribution.any_anomaly ?? 0,
                analytics?.anomaly_distribution.isolation_forest ?? 0,
                analytics?.anomaly_distribution.zscore ?? 0,
              )}
              tone="orange"
            />
            <DistributionBar
              label="Urgent / critical"
              value={analytics?.urgent_critical ?? 0}
              maxValue={analytics?.urgent_critical ?? 0}
              tone="red"
            />
          </div>
        </section>
      )}

      {analyticsAvailable && analytics && (
        <section className="content-grid analytics-grid" aria-label="Machine and severity comparison">
          <div className="panel machine-comparison-panel">
            <PanelHeading title="Machine-wise assessments" subtitle="Persisted records by asset" />
            {analytics.asset_comparison.length ? (
              (() => {
                const maxValue = Math.max(
                  ...analytics.asset_comparison.map((item) => Number.isFinite(item.count) ? Math.max(0, item.count) : 0),
                  0,
                )
                return analytics.asset_comparison.map((item) => (
                  <DistributionBar
                    key={item.asset_id}
                    label={item.asset_id}
                    value={item.count}
                    maxValue={maxValue}
                    tone="blue"
                  />
                ))
              })()
            ) : (
              <p className="form-note">No persisted assets available.</p>
            )}
          </div>
          <div className="panel severity-comparison-panel">
            <PanelHeading title="Maintenance severity" subtitle="Persisted recommendation distribution" />
            {(() => {
              const values = [
                analytics.severity_distribution.routine,
                analytics.severity_distribution.warning,
                analytics.severity_distribution.urgent,
                analytics.severity_distribution.critical,
              ]
              const maxValue = Math.max(
                ...values.map((value) => Number.isFinite(value) ? Math.max(0, value) : 0),
                0,
              )
              return (
                <>
                  <DistributionBar label="Routine" value={values[0]} maxValue={maxValue} tone="green" />
                  <DistributionBar label="Warning" value={values[1]} maxValue={maxValue} tone="orange" />
                  <DistributionBar label="Urgent" value={values[2]} maxValue={maxValue} tone="red" />
                  <DistributionBar label="Critical" value={values[3]} maxValue={maxValue} tone="red" />
                </>
              )
            })()}
          </div>
        </section>
      )}

      <section className="stat-grid" aria-label="Dataset summary">
        <StatCard label="Benchmark records" value={overview.dataset.rows.toLocaleString()} detail="AI4I 2020 samples" icon="▤" />
        <StatCard label="Failure examples" value={overview.dataset.failure_cases.toLocaleString()} detail={`${formatPercent(overview.dataset.failure_rate)} of dataset`} icon="⚠" accent="orange" />
        <StatCard label="Selected classifier" value={overview.selected_model} detail="Chosen by training CV F1" icon="⌁" accent="blue" />
        <StatCard
          label="Holdout F1 score"
          value={selectedMetrics ? formatPercent(selectedMetrics.f1) : '—'}
          detail={`${overview.dataset.test_rows.toLocaleString()} stratified test rows`}
          icon="◉"
          accent="green"
        />
      </section>

      <section className="content-grid overview-grid">
        <div className="panel model-panel">
          <PanelHeading title="Model performance" subtitle="Held-out test set · positive class is machine failure" />
          <div className="model-legend"><span /><span>F1 score</span><span className="legend-muted" /></div>
          {overview.classification_metrics.map((metric) => (
            <div className="model-row" key={metric.model}>
              <div className="model-row-title">
                <span>{metric.model}</span>
                {metric.model === overview.selected_model && <span className="selected-pill">SELECTED</span>}
              </div>
              <div className="bar-track"><div className="bar-fill" style={{ width: `${metric.f1 * 100}%` }} /></div>
              <strong>{formatPercent(metric.f1)}</strong>
            </div>
          ))}
          <div className="metric-footnote">
            Accuracy can hide missed failures because failures are rare. Compare recall, precision and F1 before choosing a model.
          </div>
        </div>

        <div className="panel feature-panel">
          <PanelHeading title="Global feature importance" subtitle="Model-level importance, not per-prediction SHAP" />
          <div className="importance-list">
            {overview.feature_importance.slice(0, 5).map((item) => (
              <div className="importance-row" key={item.feature}>
                <div className="importance-label"><span>{item.feature}</span><strong>{formatPercent(item.importance)}</strong></div>
                <div className="importance-track"><div style={{ width: `${item.importance * 100}%` }} /></div>
              </div>
            ))}
          </div>
          <div className="metric-footnote">Importance is global and does not explain an individual prediction.</div>
        </div>
      </section>

      <section className="bottom-grid">
        <div className="panel dataset-panel">
          <PanelHeading title="Dataset and validation" subtitle="Experiment setup" />
          <div className="data-line"><span>Source</span><strong>{overview.dataset.source}</strong></div>
          <div className="data-line"><span>Data type</span><strong>Synthetic benchmark dataset</strong></div>
          <div className="data-line"><span>Evaluation split</span><strong>{overview.dataset.split}</strong></div>
          <div className="data-line"><span>Features</span><strong>5 sensor fields + product type</strong></div>
        </div>
        <div className="panel anomaly-summary">
          <div className="anomaly-summary-icon">⌖</div>
          <div className="anomaly-copy">
            <span className="eyebrow">ISOLATION FOREST</span>
            <h3>{formatPercent(overview.anomaly_metrics.recall)} <span>recall on holdout</span></h3>
            <p>Flags unusual rows at a fixed 3% contamination assumption.</p>
          </div>
          <div className="anomaly-tag">Experimental</div>
        </div>
      </section>
    </>
  )
}

function DistributionBar({
  label,
  value,
  tone,
  maxValue,
}: {
  label: string
  value: number
  tone: 'orange' | 'green' | 'blue' | 'red'
  maxValue: number
}) {
  const safeValue = Number.isFinite(value) ? Math.max(0, value) : 0
  const safeMax = Number.isFinite(maxValue) ? Math.max(0, maxValue) : 0
  const percentage = safeMax > 0 ? (safeValue / safeMax) * 100 : 0
  const roundedPercentage = Math.min(100, Math.max(0, percentage))
  return (
    <div className="distribution-bar">
      <div><span>{label}</span><strong>{safeValue.toLocaleString()}</strong></div>
      <div
        className="distribution-track"
        role="progressbar"
        aria-label={`${label}: ${safeValue.toLocaleString()}`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Number(roundedPercentage.toFixed(1))}
      >
        <div className={tone} style={{ width: `${roundedPercentage}%` }} />
      </div>
    </div>
  )
}

function StatCard({
  label,
  value,
  detail,
  icon,
  accent = 'purple',
}: {
  label: string
  value: string
  detail: string
  icon: string
  accent?: string
}) {
  return (
    <div className="stat-card">
      <div className={`stat-icon ${accent}`}>{icon}</div>
      <span className="stat-label">{label}</span>
      <strong className="stat-value">{value}</strong>
      <span className="stat-detail">{detail}</span>
    </div>
  )
}

function PanelHeading({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="panel-heading">
      <div><h3>{title}</h3><p>{subtitle}</p></div>
      <span className="panel-menu">···</span>
    </div>
  )
}

function SensorForm({
  reading,
  submitting,
  buttonLabel,
  onUpdateNumber,
  onUpdateType,
  onSubmit,
}: {
  reading: Reading
  submitting: boolean
  buttonLabel: string
  onUpdateNumber: (key: Exclude<keyof Reading, 'product_type'>, value: string) => void
  onUpdateType: (value: Reading['product_type']) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
}) {
  return (
    <form className="sensor-form" onSubmit={onSubmit}>
      <label className="form-field">
        <span>Product quality type</span>
        <select value={reading.product_type} onChange={(event) => onUpdateType(event.target.value as Reading['product_type'])}>
          <option value="L">L · Low</option>
          <option value="M">M · Medium</option>
          <option value="H">H · High</option>
        </select>
      </label>
      {numericInputs.map((input) => (
        <label className="form-field" key={input.key}>
          <span>{input.label} <small>{input.unit}</small></span>
          <input
            type="number"
            value={reading[input.key]}
            min={input.min}
            max={input.max}
            step={input.step}
            required
            onChange={(event) => onUpdateNumber(input.key, event.target.value)}
          />
        </label>
      ))}
      <button className="primary-button form-submit" type="submit" disabled={submitting}>
        {submitting ? 'Analyzing…' : buttonLabel}
        {!submitting && <span>→</span>}
      </button>
      <p className="form-note">Inputs are interpreted using AI4I benchmark units. This is a research demo, not an industrial control.</p>
    </form>
  )
}

function PredictPage({
  reading,
  prediction,
  explanation,
  explanationError,
  explaining,
  hasPredictionReading,
  submitting,
  onUpdateNumber,
  onUpdateType,
  onSubmit,
  onExplain,
  onReset,
}: {
  reading: Reading
  prediction: Prediction | null
  explanation: PredictionExplanation | null
  explanationError: string | null
  explaining: boolean
  hasPredictionReading: boolean
  submitting: boolean
  onUpdateNumber: (key: Exclude<keyof Reading, 'product_type'>, value: string) => void
  onUpdateType: (value: Reading['product_type']) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
  onExplain: () => void
  onReset: () => void
}) {
  return (
    <>
      <div className="workspace-grid">
        <section className="panel form-panel">
          <PanelHeading title="Sensor operating conditions" subtitle="Enter one benchmark-shaped observation" />
          <SensorForm
            reading={reading}
            submitting={submitting || explaining}
            buttonLabel="Estimate failure score"
            onUpdateNumber={onUpdateNumber}
            onUpdateType={onUpdateType}
            onSubmit={onSubmit}
          />
        </section>
        <section className="panel result-panel">
          <PanelHeading title="Assessment" subtitle="Output from the selected classifier" />
          {prediction ? (
            <div className="result-content">
              <div className={`result-status ${prediction.priority.toLowerCase()}`}>
                <span className="result-status-dot" />
                {prediction.priority} priority
              </div>
              <div className="score-number">{formatPercent(prediction.failure_score)}</div>
              <div className="score-label">Estimated failure score</div>
              <div className="score-track"><div style={{ width: `${prediction.failure_score * 100}%` }} /></div>
              <div className="prediction-verdict">
                <strong>{prediction.predicted_failure ? 'Failure signal above threshold' : 'No failure signal above threshold'}</strong>
                <span>Classifier threshold: 50% · {prediction.model}</span>
              </div>
              <div className="notice-box">This is a model score on synthetic benchmark data, not a calibrated probability or maintenance instruction.</div>
              <div className="prediction-actions">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={onExplain}
                  disabled={!hasPredictionReading || submitting || explaining}
                >
                  {explaining ? 'Explaining…' : 'Explain Prediction'}
                </button>
                <button
                  type="button"
                  className="text-button"
                  onClick={onReset}
                  disabled={explaining}
                >
                  Clear result
                </button>
              </div>
            </div>
          ) : (
            <div className="result-empty">
              <div className="empty-gauge">⌁</div>
              <h3>Ready to assess</h3>
              <p>Submit sensor conditions to see the benchmark model output.</p>
            </div>
          )}
        </section>
      </div>
      <ExplanationPanel
        explanation={explanation}
        error={explanationError}
        loading={explaining}
        hasPrediction={prediction !== null}
        onRetry={onExplain}
      />
    </>
  )
}

function ExplanationPanel({
  explanation,
  error,
  loading,
  hasPrediction,
  onRetry,
}: {
  explanation: PredictionExplanation | null
  error: string | null
  loading: boolean
  hasPrediction: boolean
  onRetry: () => void
}) {
  const contributions = explanation
    ? [...explanation.contributions].sort(
      (left, right) => Math.abs(right.shap_value) - Math.abs(left.shap_value),
    )
    : []
  const maxMagnitude = Math.max(0, ...contributions.map((item) => Math.abs(item.shap_value)))

  return (
    <section className="panel explanation-panel" aria-live="polite" aria-busy={loading}>
      <PanelHeading title="Prediction explanation" subtitle="Local SHAP contributions for this reading" />
      {loading ? (
        <div className="explanation-state" role="status">
          <span className="spinner" />
          Computing SHAP contributions for the prediction…
        </div>
      ) : error ? (
        <div className="explanation-error" role="alert">
          <span>{error}</span>
          <button type="button" className="secondary-button" onClick={onRetry}>Retry explanation</button>
        </div>
      ) : !hasPrediction ? (
        <div className="explanation-state">
          Run a failure prediction first to request its SHAP explanation.
        </div>
      ) : !explanation ? (
        <div className="explanation-state">
          Select <strong>Explain Prediction</strong> to see which input features raise or lower the model score.
        </div>
      ) : (
        <>
          <div className="explanation-summary">
            <div>
              <span>Actual failure score (class 1)</span>
              <strong>{formatPercent(explanation.failure_score)}</strong>
            </div>
            <div>
              <span>Baseline score</span>
              <strong>{formatPercent(explanation.baseline_value)}</strong>
            </div>
          </div>
          <div className="explanation-legend" aria-label="Contribution legend">
            <span><i className="legend-increase" /> Increases failure score</span>
            <span><i className="legend-decrease" /> Decreases failure score</span>
          </div>
          <div className="explanation-contributions">
            {contributions.map((item) => {
              const magnitude = Math.abs(item.shap_value)
              const barWidth = maxMagnitude > 0 ? (magnitude / maxMagnitude) * 100 : 0
              const tone = item.shap_value > 0
                ? 'increase'
                : item.shap_value < 0
                  ? 'decrease'
                  : 'neutral'
              const direction = tone === 'increase'
                ? 'Increases'
                : tone === 'decrease'
                  ? 'Decreases'
                  : 'No change'
              const valueText = `${item.shap_value > 0 ? '+' : item.shap_value < 0 ? '-' : ''}${(magnitude * 100).toFixed(2)} pp`

              return (
                <div className="explanation-contribution-row" key={item.feature}>
                  <div className="explanation-feature">
                    <strong>{item.feature}</strong>
                    <span>Input: {formatContributionInput(item)}</span>
                  </div>
                  <div
                    className="contribution-chart"
                    role="img"
                    aria-label={`${item.feature}: ${direction.toLowerCase()} failure score by ${valueText}`}
                  >
                    <div className="contribution-negative">
                      {item.shap_value < 0 && (
                        <span className="contribution-fill decrease" style={{ width: `${barWidth}%` }} />
                      )}
                    </div>
                    <span className="contribution-zero" />
                    <div className="contribution-positive">
                      {item.shap_value > 0 && (
                        <span className="contribution-fill increase" style={{ width: `${barWidth}%` }} />
                      )}
                    </div>
                  </div>
                  <div className={`contribution-value ${tone}`}>
                    <span>{direction}</span>
                    <strong>{valueText}</strong>
                  </div>
                </div>
              )
            })}
          </div>
          <p className="explanation-note">
            {explanation.interpretation} Contributions are relative to the baseline, not causal effects; the input values shown are those used for this prediction.
          </p>
        </>
      )}
    </section>
  )
}

function formatContributionInput(item: PredictionExplanation['contributions'][number]) {
  if (typeof item.value !== 'number') {
    return item.value
  }
  const unit = numericInputs.find((input) => input.label === item.feature)?.unit
  const formatted = item.value.toLocaleString(undefined, { maximumFractionDigits: 2 })
  return unit ? `${formatted} ${unit}` : formatted
}

function EvaluationPage({ overview }: { overview: Overview }) {
  return (
    <>
      <section className="panel evaluation-intro">
        <div>
          <span className="eyebrow">EXPERIMENT 01 · BINARY CLASSIFICATION</span>
          <h2>Failure classification benchmark</h2>
          <p>Models are compared on the same stratified holdout set. The model used for predictions is selected by mean F1 across three training folds.</p>
        </div>
        <div className="split-badge"><strong>80 / 20</strong><span>train / test</span></div>
      </section>
      <section className="panel table-panel">
        <PanelHeading title="Held-out performance" subtitle={`${overview.dataset.test_rows.toLocaleString()} rows · class 1 = failure`} />
        <div className="table-scroll">
          <table>
            <thead><tr><th>Algorithm</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 score</th><th>ROC-AUC</th><th>CV F1</th></tr></thead>
            <tbody>
              {overview.classification_metrics.map((metric) => (
                <tr key={metric.model}>
                  <td><strong>{metric.model}</strong>{metric.model === overview.selected_model && <span className="selected-pill">SELECTED</span>}</td>
                  <td>{formatPercent(metric.accuracy)}</td>
                  <td>{formatPercent(metric.precision)}</td>
                  <td>{formatPercent(metric.recall)}</td>
                  <td className="table-highlight">{formatPercent(metric.f1)}</td>
                  <td>{formatPercent(metric.roc_auc)}</td>
                  <td>{formatPercent(overview.cross_validation_f1[metric.model] ?? 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="metric-footnote">The holdout set is used for reporting only. F1 from training-fold cross-validation selects the operating classifier.</div>
      </section>
      <section className="content-grid evaluation-lower">
        {overview.classification_metrics.map((metric) => (
          <div className="panel confusion-panel" key={metric.model}>
            <PanelHeading title={metric.model} subtitle="Confusion matrix · actual rows × predicted columns" />
            <div className="confusion-grid">
              <div className="matrix-label" />
              <div className="matrix-label">Predicted: healthy</div>
              <div className="matrix-label">Predicted: failure</div>
              <div className="matrix-label">Actual: healthy</div>
              <div className="matrix-cell good">{metric.confusion_matrix.true_negative}<small>True negative</small></div>
              <div className="matrix-cell warn">{metric.confusion_matrix.false_positive}<small>False positive</small></div>
              <div className="matrix-label">Actual: failure</div>
              <div className="matrix-cell alert">{metric.confusion_matrix.false_negative}<small>False negative</small></div>
              <div className="matrix-cell good">{metric.confusion_matrix.true_positive}<small>True positive</small></div>
            </div>
          </div>
        ))}
      </section>
    </>
  )
}

function AnomalyPage({
  reading,
  anomaly,
  zscore,
  metrics,
  submitting,
  onUpdateNumber,
  onUpdateType,
  onSubmit,
  onSubmitZscore,
}: {
  reading: Reading
  anomaly: AnomalyResult | null
  zscore: ZscoreResult | null
  metrics: Overview['anomaly_metrics']
  submitting: boolean
  onUpdateNumber: (key: Exclude<keyof Reading, 'product_type'>, value: string) => void
  onUpdateType: (value: Reading['product_type']) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
  onSubmitZscore: () => void
}) {
  return (
    <>
      <section className="evaluation-intro anomaly-intro">
        <div>
          <span className="eyebrow">EXPERIMENT 02 · ISOLATION FOREST</span>
          <h2>Operating-condition outlier check</h2>
          <p>The detector is fit on training sensor features only. Its 3% contamination setting is an operating assumption, not a learned failure score.</p>
        </div>
        <div className="mini-metrics">
          <div><strong>{formatPercent(metrics.precision)}</strong><span>holdout precision</span></div>
          <div><strong>{formatPercent(metrics.recall)}</strong><span>holdout recall</span></div>
          <div><strong>{formatPercent(metrics.f1)}</strong><span>holdout F1</span></div>
        </div>
      </section>
      <div className="workspace-grid">
        <section className="panel form-panel">
          <PanelHeading title="Sensor operating conditions" subtitle="Use the same benchmark fields" />
          <SensorForm
            reading={reading}
            submitting={submitting}
            buttonLabel="Check for anomaly"
            onUpdateNumber={onUpdateNumber}
            onUpdateType={onUpdateType}
            onSubmit={onSubmit}
          />
        </section>
        <section className="panel result-panel">
          <PanelHeading title="Outlier assessment" subtitle="Isolation Forest result" />
          {anomaly ? (
            <div className="result-content">
              <div className={`result-status ${anomaly.is_anomaly ? 'critical' : 'low'}`}>
                <span className="result-status-dot" />
                {anomaly.is_anomaly ? 'Unusual pattern flagged' : 'No unusual pattern flagged'}
              </div>
              <div className="score-number score-text">{anomaly.is_anomaly ? 'Flagged' : 'Within profile'}</div>
              <div className="score-label">Raw anomaly score: {anomaly.anomaly_score.toFixed(4)}</div>
              <div className="notice-box">Anomaly score is not a probability. A flagged row is not proof of failure.</div>
            </div>
          ) : (
            <div className="result-empty">
              <div className="empty-gauge">⌖</div>
              <h3>Ready to inspect</h3>
              <p>Run the detector to compare this reading with the training feature distribution.</p>
            </div>
          )}
        </section>
      </div>
      <section className="panel zscore-panel">
        <PanelHeading title="Z-score statistical detection" subtitle="Healthy training observations define the numeric feature baseline" />
        <button className="secondary-button" type="button" onClick={onSubmitZscore} disabled={submitting}>
          {submitting ? 'Analyzing…' : 'Run Z-score detection'}
        </button>
        {zscore ? (
          <div className="zscore-result">
            <div className={`result-status ${zscore.is_anomaly ? 'critical' : 'low'}`}>
              <span className="result-status-dot" />
              {zscore.is_anomaly ? 'Anomaly indicated by Z-score' : 'No Z-score anomaly indicated'}
            </div>
            <div className="score-label">Maximum absolute Z-score: {zscore.max_absolute_zscore.toFixed(3)} · Threshold: ≥ {zscore.threshold.toFixed(1)}</div>
            <div className="zscore-list">
              {Object.entries(zscore.feature_zscores).map(([feature, value]) => {
                const exceeded = zscore.exceeded_features.includes(feature)
                return (
                  <div className={`zscore-row ${exceeded ? 'exceeded' : ''}`} key={feature}>
                    <span>{feature}</span>
                    <strong>{value >= 0 ? '+' : ''}{value.toFixed(3)}</strong>
                    {exceeded && <small>Exceeds threshold</small>}
                  </div>
                )
              })}
            </div>
            <div className="notice-box">Z-score anomaly status indicates statistical deviation, not proof of equipment failure.</div>
          </div>
        ) : (
          <p className="form-note">Run Z-score detection against the same sensor reading to compare it with the healthy training baseline.</p>
        )}
      </section>
      <section className="panel method-note">
        <strong>Evaluation note</strong>
        <span>Holdout labels are used to report precision, recall and F1, but are not used to fit Isolation Forest. The fixed contamination assumption affects how many readings are flagged.</span>
      </section>
    </>
  )
}

function SimulationPage({
  status,
  history,
  onStart,
  onStop,
  onGenerate,
  onClear,
  submitting,
}: {
  status: SimulationStatus | null
  history: SimulationEvent[]
  onStart: () => void
  onStop: () => void
  onGenerate: (mode: 'normal' | 'abnormal' | 'mixed') => void
  onClear: () => void
  submitting: boolean
}) {
  const latest = status?.latest ?? history[0] ?? null
  return (
    <>
      <section className="evaluation-intro anomaly-intro">
        <div>
          <span className="eyebrow">SIMULATION ONLY · NO REAL TELEMETRY</span>
          <h2>Simulated sensor monitoring</h2>
          <p>Changing AI4I-shaped readings are generated locally for demonstration. They are not measurements from industrial equipment.</p>
        </div>
        <div className="mini-metrics">
          <div><strong>{status?.simulator_running ? 'Running' : 'Paused'}</strong><span>monitoring status</span></div>
          <div><strong>{status?.samples_published ?? 0}</strong><span>simulated samples</span></div>
          <div><strong>{status?.interval_seconds ?? 2}s</strong><span>poll interval</span></div>
        </div>
      </section>

      <div className="simulation-actions">
        <button className="primary-button" type="button" onClick={onStart} disabled={submitting || status?.simulator_running}>Start / resume</button>
        <button className="secondary-button" type="button" onClick={onStop} disabled={submitting || !status?.simulator_running}>Pause</button>
        <button className="secondary-button" type="button" onClick={() => onGenerate('normal')} disabled={submitting}>Generate normal</button>
        <button className="secondary-button" type="button" onClick={() => onGenerate('abnormal')} disabled={submitting}>Generate abnormal</button>
        <button className="text-button" type="button" onClick={onClear} disabled={submitting}>Clear history</button>
      </div>

      {latest ? (
        <div className="workspace-grid">
          <section className="panel form-panel">
            <PanelHeading title="Latest simulated reading" subtitle={`${latest.simulation_mode ?? 'mixed'} mode · local synthetic data`} />
            <div className="simulation-readings">
              {[
                ['Product type', latest.reading.product_type],
                ['Air temperature', `${latest.reading.air_temperature.toFixed(2)} K`],
                ['Process temperature', `${latest.reading.process_temperature.toFixed(2)} K`],
                ['Rotational speed', `${latest.reading.rotational_speed.toFixed(2)} rpm`],
                ['Torque', `${latest.reading.torque.toFixed(2)} Nm`],
                ['Tool wear', `${latest.reading.tool_wear.toFixed(2)} min`],
              ].map(([label, value]) => (
                <div className="simulation-reading" key={label}><span>{label}</span><strong>{value}</strong></div>
              ))}
            </div>
          </section>
          <section className="panel result-panel">
            <PanelHeading title="Assessment from simulated reading" subtitle="AI4I failure and anomaly models" />
            <div className="simulation-assessment">
              <div>
                <span>Failure prediction</span>
                <strong>{formatPercent(latest.prediction.failure_score)}</strong>
                <small>{latest.prediction.model} · {latest.prediction.priority}</small>
              </div>
              <div>
                <span>Isolation Forest</span>
                <strong>{latest.anomaly.is_anomaly ? 'Anomaly' : 'Normal'}</strong>
                <small>Score {latest.anomaly.anomaly_score.toFixed(4)}</small>
              </div>
              <div>
                <span>Z-score</span>
                <strong>{latest.zscore.is_anomaly ? 'Anomaly' : 'Normal'}</strong>
                <small>Maximum {latest.zscore.max_absolute_zscore.toFixed(3)}</small>
              </div>
              {latest.recommendation && (
                <div className={`recommendation-severity ${latest.recommendation.severity}`}>
                  <span>Maintenance severity</span>
                  <strong>{latest.recommendation.severity.toUpperCase()}</strong>
                  {(latest.recommendation.reasons ?? []).map((reason) => (
                    <small key={reason.code}>{reason.message}</small>
                  ))}
                  {(latest.recommendation.actions ?? []).length > 0 && (
                    <div className="recommendation-actions">
                      <strong>Recommended actions</strong>
                      <ul>
                        {(latest.recommendation.actions ?? []).map((action) => (
                          <li key={action.code}>
                            <strong>{action.title}</strong>
                            <span>{action.detail}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          </section>
        </div>
      ) : (
        <div className="empty-panel">
          <h2>No simulated sample yet</h2>
          <p>Start monitoring or generate a normal or abnormal simulated reading.</p>
        </div>
      )}

      <section className="panel table-panel">
        <PanelHeading title="Recent simulated readings" subtitle="Most recent entries shown first · bounded to 100 backend samples" />
        {history.length ? (
          <div className="table-scroll">
            <table>
              <thead><tr><th>Mode</th><th>Air</th><th>Process</th><th>Speed</th><th>Torque</th><th>Wear</th><th>Failure</th><th>IF</th><th>Z-score</th></tr></thead>
              <tbody>
                {history.map((item, index) => (
                  <tr key={`${item.created_at ?? 'sim'}-${index}`}>
                    <td><strong>{item.simulation_mode ?? 'mixed'}</strong></td>
                    <td>{item.reading.air_temperature.toFixed(1)}</td>
                    <td>{item.reading.process_temperature.toFixed(1)}</td>
                    <td>{item.reading.rotational_speed.toFixed(0)}</td>
                    <td>{item.reading.torque.toFixed(1)}</td>
                    <td>{item.reading.tool_wear.toFixed(1)}</td>
                    <td>{formatPercent(item.prediction.failure_score)}</td>
                    <td>{item.anomaly.is_anomaly ? 'Anomaly' : 'Normal'}</td>
                    <td>{item.zscore.is_anomaly ? 'Anomaly' : 'Normal'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="form-note">No simulated history available.</p>
        )}
      </section>
    </>
  )
}

function AssessmentHistoryPage({
  meta,
  items,
  onLoadMore,
  assetFilter,
  kindFilter,
  startFilter,
  endFilter,
  onAssetFilter,
  onKindFilter,
  onStartFilter,
  onEndFilter,
  onApplyFilters,
}: {
  meta: AssessmentHistoryResponse | null
  items: PersistedAssessment[]
  onLoadMore: (offset: number) => void
  assetFilter: string
  kindFilter: string
  startFilter: string
  endFilter: string
  onAssetFilter: (value: string) => void
  onKindFilter: (value: string) => void
  onStartFilter: (value: string) => void
  onEndFilter: (value: string) => void
  onApplyFilters: () => void
}) {
  const offset = meta?.offset ?? 0
  const limit = meta?.limit ?? 50
  const nextOffset = offset + limit
  const previousOffset = Math.max(0, offset - limit)
  return (
    <>
      <section className="evaluation-intro anomaly-intro">
        <div>
          <span className="eyebrow">PERSISTED DATABASE RECORDS</span>
          <h2>Assessment history</h2>
          <p>Historical records are loaded only from PostgreSQL or the configured database. Simulated monitoring history remains separate and in memory.</p>
        </div>
      </section>
      {meta && !meta.database_connected ? (
        <div className="empty-panel">
          <h2>Database unavailable</h2>
          <p>Assessment history could not be loaded. Current simulation and benchmark features remain available.</p>
        </div>
      ) : (
        <section className="panel table-panel">
          <div className="history-filter-row">
            <label>
              <span>Asset ID</span>
              <input value={assetFilter} onChange={(event) => onAssetFilter(event.target.value)} placeholder="All assets" />
            </label>
            <label>
              <span>Assessment type</span>
              <select value={kindFilter} onChange={(event) => onKindFilter(event.target.value)}>
                <option value="">All types</option>
                <option value="machine_assessment">Machine assessment</option>
                <option value="failure_prediction">Failure prediction</option>
                <option value="anomaly_check">Anomaly check</option>
                <option value="rul_estimate">RUL estimate</option>
              </select>
            </label>
            <label>
              <span>Start date</span>
              <input type="datetime-local" value={startFilter} onChange={(event) => onStartFilter(event.target.value)} />
            </label>
            <label>
              <span>End date</span>
              <input type="datetime-local" value={endFilter} onChange={(event) => onEndFilter(event.target.value)} />
            </label>
            <button className="secondary-button" type="button" onClick={onApplyFilters}>Apply filters</button>
          </div>
          <PanelHeading
            title="Recent assessments"
            subtitle={meta ? `${meta.total.toLocaleString()} total matching records` : 'Loading persisted records'}
          />
          {items.length ? (
            <div className="table-scroll">
              <table>
                <thead><tr><th>ID</th><th>Timestamp</th><th>Asset</th><th>Source</th><th>Kind</th><th>Sensor / assessment</th><th>Record</th></tr></thead>
                <tbody>
                  {items.map((item) => (
                    <tr key={item.id}>
                      <td><strong>{item.id}</strong></td>
                      <td>{item.created_at ? new Date(item.created_at).toLocaleString() : '—'}</td>
                      <td>{item.asset_id}</td>
                      <td>{item.source}</td>
                      <td>{item.kind}</td>
                      <td>
                        {typeof item.reading.torque === 'number'
                          ? `Torque ${Number(item.reading.torque).toFixed(1)} Nm`
                          : 'Sensor record'}
                        {item.prediction && typeof item.prediction.failure_score === 'number'
                          ? ` · Failure ${(Number(item.prediction.failure_score) * 100).toFixed(1)}%`
                          : ''}
                      </td>
                      <td>{item.simulated ? 'Simulated · persisted' : 'Persistent assessment'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="form-note">No persisted assessment records are available.</p>
          )}
          {meta && (
            <div className="simulation-actions">
              <button className="secondary-button" type="button" disabled={offset === 0} onClick={() => onLoadMore(previousOffset)}>Previous</button>
              <button className="secondary-button" type="button" disabled={nextOffset >= (meta.total ?? 0)} onClick={() => onLoadMore(nextOffset)}>Next</button>
            </div>
          )}
        </section>
      )}
    </>
  )
}

function RulPage({
  overview,
  loading,
  error,
  reading,
  prediction,
  submitting,
  onUpdate,
  onSubmit,
}: {
  overview: RulOverview | null
  loading: boolean
  error: string | null
  reading: RulReading
  prediction: RulPrediction | null
  submitting: boolean
  onUpdate: (key: keyof RulReading, value: string) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
}) {
  const rulFields = (Object.keys(reading) as (keyof RulReading)[])
  const isSensor = (key: keyof RulReading) => String(key).startsWith('sensor_')

  return (
    <>
      <section className="evaluation-intro anomaly-intro">
        <div>
          <span className="eyebrow">NASA C-MAPSS · FD001 · ENGINE LIFECYCLE</span>
          <h2>Remaining useful life estimation</h2>
          <p>Separate from AI4I failure classification: the selected model estimates remaining operating cycles for a turbine engine snapshot.</p>
        </div>
        {overview && (
          <div className="mini-metrics">
            <div><strong>{overview.dataset.training_engines}</strong><span>training engines</span></div>
            <div><strong>{overview.dataset.test_engines}</strong><span>test engines</span></div>
            <div><strong>{overview.dataset.rul_cap}</strong><span>cycle cap</span></div>
          </div>
        )}
      </section>

      {overview ? (
        <>
          <section className="panel table-panel">
            <PanelHeading title="RUL model comparison" subtitle="Training engine GroupKFold validation and official FD001 test engines" />
            <div className="table-scroll">
              <table>
                <thead><tr><th>Algorithm</th><th>Validation RMSE</th><th>Validation MAE</th><th>Validation NASA</th><th>Test RMSE</th><th>Test MAE</th><th>Test NASA</th></tr></thead>
                <tbody>
                  {overview.test_metrics.map((testMetric) => {
                    const validationMetric = overview.validation_metrics.find((item) => item.model === testMetric.model)
                    return (
                      <tr key={testMetric.model}>
                        <td><strong>{testMetric.model}</strong>{testMetric.model === overview.selected_model && <span className="selected-pill">SELECTED</span>}</td>
                        <td>{validationMetric?.rmse.toFixed(3) ?? '—'}</td>
                        <td>{validationMetric?.mae.toFixed(3) ?? '—'}</td>
                        <td>{validationMetric?.nasa_score.toFixed(3) ?? '—'}</td>
                        <td className="table-highlight">{testMetric.rmse.toFixed(3)}</td>
                        <td>{testMetric.mae.toFixed(3)}</td>
                        <td>{testMetric.nasa_score.toFixed(3)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <div className="metric-footnote">
              Final test metrics use only the last observation of each test engine. Lower RMSE, MAE, and NASA scores are better.
            </div>
          </section>

          <div className="workspace-grid">
            <section className="panel form-panel">
              <PanelHeading title="Engine snapshot" subtitle="3 operational settings and 21 FD001 sensor measurements" />
              <form className="sensor-form rul-sensor-form" onSubmit={onSubmit}>
                {rulFields.map((key) => (
                  <label className="form-field" key={key}>
                    <span>{isSensor(key) ? `Sensor ${String(key).replace('sensor_', '')}` : String(key).replaceAll('_', ' ')}</span>
                    <input
                      type="number"
                      value={reading[key]}
                      step="any"
                      required
                      onChange={(event) => onUpdate(key, event.target.value)}
                    />
                  </label>
                ))}
                <button className="primary-button form-submit" type="submit" disabled={submitting}>
                  {submitting ? 'Estimating…' : 'Estimate remaining useful life'}
                </button>
              </form>
            </section>

            <section className="panel result-panel">
              <PanelHeading title="RUL assessment" subtitle="Selected NASA C-MAPSS regressor" />
              {prediction ? (
                <div className="result-content">
                  <div className="score-number">{prediction.predicted_rul.toFixed(1)}</div>
                  <div className="score-label">Estimated remaining cycles</div>
                  <div className="notice-box">
                    <strong>{prediction.model}</strong><br />
                    {prediction.interpretation}
                  </div>
                </div>
              ) : (
                <div className="result-empty">
                  <div className="empty-gauge">◷</div>
                  <h3>Ready to estimate</h3>
                  <p>Submit an FD001-shaped engine snapshot to estimate remaining cycles.</p>
                </div>
              )}
            </section>
          </div>
        </>
      ) : (
        <div className="empty-panel" role="status" aria-live="polite">
          {loading ? (
            <>
              <h2>Loading NASA RUL models</h2>
              <p>The first request trains the FD001 models. Keep the backend running while it initializes.</p>
            </>
          ) : (
            <>
              <h2>NASA RUL model not available</h2>
              <p>{error ?? 'Check the backend response and confirm the FD001 dataset is available.'}</p>
            </>
          )}
        </div>
      )}
    </>
  )
}

function AboutPage({ overview }: { overview: Overview }) {
  return (
    <div className="about-grid">
      <section className="panel about-main">
        <span className="eyebrow">PROJECT SCOPE</span>
        <h2>What this prototype does</h2>
        <p>IntelliMaint is a research prototype for evaluating predictive-maintenance methods on public benchmark data. It compares binary failure classifiers and demonstrates a separate anomaly-detection workflow.</p>
        <div className="scope-list">
          <div><span className="check-mark">✓</span><span><strong>Failure classification</strong><small>Logistic Regression, Random Forest, Decision Tree and XGBoost on AI4I 2020.</small></span></div>
          <div><span className="check-mark">✓</span><span><strong>Model evaluation</strong><small>Stratified holdout metrics and training-fold model selection.</small></span></div>
          <div><span className="check-mark">✓</span><span><strong>Anomaly detection</strong><small>Isolation Forest with an explicit 3% contamination assumption.</small></span></div>
          <div><span className="check-mark">✓</span><span><strong>NASA C-MAPSS RUL</strong><small>FD001 engine-life regression is available as a separate benchmark pipeline.</small></span></div>
          <div><span className="check-mark">✓</span><span><strong>Per-reading SHAP explanations</strong><small>Local contributions explain the selected classifier's uncalibrated class-1 score.</small></span></div>
        </div>
      </section>
      <section className="panel limitations-panel">
        <span className="eyebrow">READ BEFORE DEMO</span>
        <h3>Limitations</h3>
        <ul>
          <li>AI4I is a synthetic benchmark, not live industrial telemetry.</li>
          <li>Failure score is not calibrated and must not control equipment.</li>
          <li>The test results apply to this benchmark and split only.</li>
          <li>Recommendations are illustrative triage labels, not maintenance advice.</li>
          <li>RUL, database persistence and real-time machine integration are future work.</li>
        </ul>
        <div className="dataset-credit">
          <strong>Dataset</strong>
          <span>{overview.dataset.name}</span>
          <span>UCI Machine Learning Repository · DOI 10.24432/C5HS5C · CC BY 4.0</span>
        </div>
      </section>
    </div>
  )
}

export default App
