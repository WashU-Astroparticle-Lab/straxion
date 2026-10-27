# Data structure

Arrows point from a product to its direct dependencies: `A --> B` means
“A depends on B.” Node colors indicate data kinds. Products with different data
kinds require separate `get_df` (or `get_array`) calls.

## Offline processing

The `qualiphide_thz_offline` context uses the following dependency graph.
`DxHitClassification` consumes `hits` and `noises`; optimal-filter fitting and
noise PSD estimation remain in that plugin. `SubstrateEvents` consumes only
`hit_classification`.

```mermaid
flowchart BT
    raw_records["raw_records<br/>QUALIPHIDETHzReader"]
    truth["truth<br/>Truth"]
    records["records<br/>DxRecords"]
    hits["hits<br/>DxHits"]
    noises["noises<br/>NoiseBank"]
    hit_classification["hit_classification<br/>DxHitClassification"]
    substrate_events["substrate_events<br/>SubstrateEvents"]
    match["match<br/>Match"]

    truth --> raw_records
    records --> raw_records
    records --> truth
    hits --> records
    noises --> records
    noises --> hits
    hit_classification --> hits
    hit_classification --> noises
    substrate_events --> hit_classification
    match --> truth
    match --> hits
    match --> hit_classification

    classDef raw fill:#dbeafe,stroke:#1d4ed8,color:#172554
    classDef records fill:#ccfbf1,stroke:#0f766e,color:#134e4a
    classDef hits fill:#fef3c7,stroke:#b45309,color:#78350f
    classDef noises fill:#ede9fe,stroke:#6d28d9,color:#4c1d95
    classDef truth fill:#fce7f3,stroke:#be185d,color:#831843
    classDef events fill:#dcfce7,stroke:#15803d,color:#14532d
    class raw_records raw
    class records records
    class hits,hit_classification hits
    class noises noises
    class truth,match truth
    class substrate_events events
```

Moving-average peak coincidence and native peak/support metadata are produced
with `hits` and carried into `hit_classification`. The optional legacy
record-threshold coincidence is computed upstream in `DxHits`. See the
[peak timing contract](../../docs/PEAK_CONTRACT.md) for selection defaults,
legacy comparisons and cache migration.

## Online and test processing

`qualiphide_thz_online` uses the theta pipeline below. The `qualiphide` test
context has the same dependencies, with `NX3LikeReader` providing `raw_records`.

```mermaid
flowchart BT
    raw_records["raw_records<br/>QUALIPHIDETHzReader or NX3LikeReader"]
    records["records<br/>PulseProcessing"]
    hits["hits<br/>Hits"]
    hit_classification["hit_classification<br/>HitClassification"]
    baseline_monitor["baseline_monitor<br/>BaselineMonitor"]

    records --> raw_records
    hits --> records
    hit_classification --> hits
    baseline_monitor --> records

    classDef raw fill:#dbeafe,stroke:#1d4ed8,color:#172554
    classDef records fill:#ccfbf1,stroke:#0f766e,color:#134e4a
    classDef hits fill:#fef3c7,stroke:#b45309,color:#78350f
    class raw_records raw
    class records,baseline_monitor records
    class hits,hit_classification hits
```

## Data kinds

| Data kind | Products | Color |
| --- | --- | --- |
| `raw_records` | `raw_records` | Blue |
| `records` | `records`, `baseline_monitor` | Teal |
| `hits` | `hits`, `hit_classification` | Amber |
| `noises` | `noises` | Purple |
| `truth` | `truth`, `match` | Pink |
| `substrate_events` | `substrate_events` | Green |
