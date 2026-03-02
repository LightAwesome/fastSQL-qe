

# 🤠 Fastest SQL Query Engine in the West

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![NumPy](https://img.shields.io/badge/numpy-vectorized-013243)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-production--ready-success)

> A vectorized, rule-based SQL query engine built from scratch in Python.
> Implements parsing, analysis, logical planning, optimization, and a batched execution engine with measurable performance wins.

---

## 🧠 What This Is

This project implements a **miniature relational query engine** from first principles.

Given:

```sql
SELECT active, SUM(value)
FROM t
WHERE value > 0.999
GROUP BY active;
```

It:

1. **Tokenizes** SQL
2. **Parses** into an AST
3. **Analyzes** for semantic correctness
4. Builds a **Logical Plan**
5. Applies **rule-based optimizations**
6. Compiles into a **vectorized execution pipeline**
7. Executes in **batches using NumPy**

The goal was not feature completeness —
The goal was **architectural correctness + measurable optimization impact**.

---

## ⚙️ Architecture Overview

### Pipeline

```
SQL String
   ↓
Tokenizer
   ↓
Parser (AST)
   ↓
Analyzer (semantic validation)
   ↓
Logical Plan
   ↓
Optimizer (rule-based rewrites)
   ↓
Physical Operators (Scan → Filter → Project → Agg → ...)
   ↓
Vectorized Batch Execution
```

---

## 🏗 Core Design Decisions

### 1️⃣ Vectorized Execution

* All data stored as **NumPy arrays**
* Operators process **batches of 4096 rows**
* No Python row loops in hot paths
* Boolean masks used for filtering

---

### 2️⃣ Iterator Operator Model

Each operator implements:

```python
def batches(self) -> Iterator[dict[str, np.ndarray]]:
```

Operators compose into pipelines:

```
Scan → Filter → Project → Aggregate → Limit
```

---

### 3️⃣ Blocking vs Streaming Operators

| Operator  | Type                        |
| --------- | --------------------------- |
| Scan      | Streaming                   |
| Filter    | Streaming                   |
| Project   | Streaming                   |
| Limit     | Streaming (early stop)      |
| Sort      | Blocking                    |
| Aggregate | Blocking (hash aggregation) |
| TopN      | Blocking (optimized)        |

---

## 🚀 Optimizations Implemented

### 🔥 1. Projection Pushdown

Only required columns are scanned.

Before:

```
Scan(table=t, cols=[*])
```

After:

```
Scan(table=t, cols=[id, value])
```

Reduces memory bandwidth significantly for wide tables.

---

### ⚡ 2. Early-Stop LIMIT (Streaming)

`SELECT id FROM t LIMIT 10`

Stops scanning after the first batch instead of scanning all 10M rows.

This is implemented as a **short-circuiting operator** in the iterator pipeline.

---

### 🏎 3. Top-N Optimization (`ORDER BY ... LIMIT N`)

Rewrites:

```
Limit
  Sort
```

Into:

```
TopN
```

Uses `numpy.argpartition`:

* O(M) selection of best N
* O(N log N) final sort
* Avoids full table sort

Massive performance improvement for large datasets.

---

## 📊 Benchmark Results (10,000,000 Rows)

```
Query                                          no-opt med/p95 (ms)  opt med/p95 (ms)    speedup   opt rows/batches
-------------------------------------------------------------------------------------------------------------------
LIMIT early stop                                  15.7/   42.3            0.0/    0.0        3835.17x   4,096/1
Selective filter + projection                    384.8/  659.2          246.8/  251.7           1.56x   10,000,000/2,442
Compute expression                              1578.2/ 1617.9            0.2/    0.2        9290.56x   4,096/1
Filter + GROUP BY (optimizer impact)             417.2/  418.1          257.6/  260.4           1.62x   10,000,000/2,442
GROUP BY low cardinality (no filter)            3625.4/ 4257.5         3439.1/ 3449.1           1.05x   10,000,000/2,442
Sort + LIMIT                                    1527.8/ 1541.5          120.4/  121.9          12.69x   10,000,000/2,442
```

### Key Takeaways

* **LIMIT early-stop:** >3000× faster
* **Projection pushdown:** ~1.6× faster on wide tables
* **Top-N optimization:** ~12× faster than full sort
* GROUP BY remains blocking by design (documented tradeoff)

---

## 🧾 EXPLAIN Output

Example rewrite:

### Before Optimization

```
Limit(n=100)
  Sort(key=value)
    Project(select=[id, value])
      Scan(table=t, cols=[*])
```

### After Optimization

```
TopN(n=100)
  Project(select=[id, value])
    Scan(table=t, cols=[id, value])
```

The optimizer transforms the logical plan before execution.

---

## 🧩 Features Supported

* SELECT
* WHERE
* GROUP BY
* ORDER BY
* LIMIT
* COUNT, SUM, AVG, MIN, MAX
* Aliases (`AS`)
* Deterministic column naming
* EXPLAIN plan printing

---

## 🛠 Installation

```bash
git clone https://github.com/yourusername/fastest-sql-query-engine-in-the-west.git
cd fastest-sql-query-engine-in-the-west
pip install -r requirements.txt
```

---

## 🧪 Run Tests

```bash
python -m pytest
```

---

## 🏁 Run Benchmarks

```bash
python -m qe.bench.bench
```

---

## 🧠 What This Project Demonstrates

* Understanding of SQL compilation pipeline
* Rule-based query optimization
* Vectorized data processing
* Iterator-based execution engines
* Algorithmic performance tradeoffs
* Hash aggregation vs streaming design
* Top-N selection optimization
* Clean separation of logical vs physical planning

---

## 📌 Design Tradeoffs

### Blocking Hash Aggregation

Simple and correct.
Streaming aggregation over sorted input is possible future work.

### Single-Key ORDER BY for TopN

Keeps implementation clean and performance measurable.

### No Indexing

This is a scan-based engine. Indexes are future work.

---

## 📈 Future Work

* Streaming hash aggregation
* Multi-key Top-N
* Cost-based optimizer
* Join operator
* Disk-backed spilling
* Parallel execution
* Predicate reordering

---

## 🏆 Why This Is Interesting

This is not a wrapper around SQLite.
It is not a Pandas shortcut.

It is a full SQL execution engine with:

* A tokenizer
* A parser
* An analyzer
* A logical plan
* An optimizer
* A vectorized execution backend
* Measured optimization gains

Built from scratch.

---

## 📜 License

MIT License

---

# 🤠 Fastest SQL Query Engine in the West

Because real cowboys don’t full-sort 10 million rows when they only need 100.

---


