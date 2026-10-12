# 🎬 CineMatch: Complete End-to-End Project Documentation
## Intelligent Data Systems — Pure KNN Movie Recommendation & Analytics Engine

This document provides a comprehensive, technical, and architectural breakdown of the **CineMatch** project. The project is built entirely on **Item-Based Collaborative Filtering** powered by **K-Nearest Neighbors (KNN)** using **Cosine Similarity**, served by a high-performance **Flask REST API**, and presented through a modern **React + Vite Web Application**.

---

## 📑 Table of Contents
1. [Project Overview & High-Level Architecture](#1-project-overview--high-level-architecture)
2. [Machine Learning & Data Science Core (Pure KNN Engine)](#2-machine-learning--data-science-core-pure-knn-engine)
   - [2.1 Datasets & Schema Definition](#21-datasets--schema-definition)
   - [2.2 Data Preprocessing & Sparsity Reduction Pipeline](#22-data-preprocessing--sparsity-reduction-pipeline)
   - [2.3 Item-Based Collaborative Filtering (KNN) Algorithm](#23-item-based-collaborative-filtering-knn-algorithm)
   - [2.4 Title Normalization & Multi-Tier Search Engine](#24-title-normalization--multi-tier-search-engine)
   - [2.5 Dynamic Genre Filtering with Neighbor Expansion](#25-dynamic-genre-filtering-with-neighbor-expansion)
   - [2.6 Exploratory Data Analysis (EDA) & Descriptive Statistics](#26-exploratory-data-analysis-eda--descriptive-statistics)
   - [2.7 Offline Model Evaluation (Precision@K & Recall@K)](#27-offline-model-evaluation-precisionk--recallk)
3. [Backend Architecture (Flask REST API)](#3-backend-architecture-flask-rest-api)
   - [3.1 Application Lifecycle & In-Memory Pipeline State](#31-application-lifecycle--in-memory-pipeline-state)
   - [3.2 REST API Endpoints Specification](#32-rest-api-endpoints-specification)
   - [3.3 Error Handling & Edge Cases](#33-error-handling--edge-cases)
4. [Frontend Architecture (React + Vite Web App)](#4-frontend-architecture-react--vite-web-app)
   - [4.1 Tech Stack & UI Philosophy](#41-tech-stack--ui-philosophy)
   - [4.2 Client State & Interactive Components](#42-client-state--interactive-components)
   - [4.3 Proxy Configuration & API Integration](#43-proxy-configuration--api-integration)
5. [Complete End-to-End Data Flow](#5-complete-end-to-end-data-flow)
6. [Automated Unit Testing & Verification](#6-automated-unit-testing--verification)
7. [Execution & Run Guide](#7-execution--run-guide)
8. [Viva & Technical Interview Guide (FAQ)](#8-viva--technical-interview-guide-faq)

---

## 1. Project Overview & High-Level Architecture

**CineMatch** is an end-to-end, personalized movie recommendation system built from first principles using standard scientific Python packages (**NumPy**, **Pandas**, **SciPy**, and **Scikit-Learn**). It addresses movie discovery by modeling user preference patterns through high-dimensional item rating vectors. 

The system relies exclusively on **Item-Based Collaborative Filtering using K-Nearest Neighbors (KNN)** with **Cosine Similarity / Distance**. By analyzing how 610 users rated 2,269 qualified movies, the model calculates multidimensional geometric angles between movie vectors, discovering hidden similarities and recommending relevant titles in real time.

### System Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                                  DATA LAYER                                       |
|  - movies.csv (9,742 titles)            - ratings.csv (100,836 ratings)           |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       DATA PREPROCESSING & PIPELINE (src/)                        |
|  - Deduplication & Schema Validation                                              |
|  - Regex release year extraction: "Toy Story (1995)" -> 1995                     |
|  - Interaction filtering: Movies >= 10 ratings, Users >= 5 ratings                |
|  - Sparsity reduction: 98.30% down to 94.14%                                      |
|  - Item-User Pivot Table (2,269 Movies x 610 Users)                               |
|  - Compressed Sparse Row (CSR) matrix compression (scipy.sparse.csr_matrix)       |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           MACHINE LEARNING ENGINE (KNN)                           |
|  - CineMatchKNN: NearestNeighbors(metric='cosine', algorithm='brute')             |
|  - High-dimensional vector query: argmin (1 - cos(theta))                         |
|  - Dynamic metadata enrichment (release year, avg rating, vote count, genres)     |
|  - Offline Model Evaluator: Precision@K & Recall@K on stratified 80/20 test split |
|  - Statistical EDA: Mean, median, mode, variance, std dev, skewness, and IQR      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         BACKEND SERVICE (server.py)                               |
|  Flask REST API (Port 5000)                                                       |
|  - Global in-memory pipeline state caching (STATE)                                |
|  - Cross-Origin Resource Sharing (CORS) enabled                                   |
|  - Endpoints: /api/recommend, /api/movies, /api/stats, /api/evaluation, /api/health|
+-----------------------------------------+-----------------------------------------+
                                          |  HTTP / JSON (Vite Reverse Proxy)
                                          v
+-----------------------------------------------------------------------------------+
|                        FRONTEND CLIENT (frontend/ - React 18)                     |
|  - Vite Development Server (Port 5173)                                            |
|  - Real-time debounced autocomplete search across catalog titles                  |
|  - One-click quick-pick recommendation chips                                      |
|  - Dynamic genre filter & configurable Top-K (6, 9, 12 movies)                    |
|  - Seed movie spotlight banner showcasing user choice's vector origin             |
|  - High-precision movie cards featuring percentage match badge & star ratings     |
+-----------------------------------------------------------------------------------+
```

---

## 2. Machine Learning & Data Science Core (Pure KNN Engine)

The machine learning core is organized modularly under `src/`:
- `src/data_processing.py`: Ingestion, schema validation, data cleaning, sparsity filtering, CSR compression.
- `src/recommender.py`: Item-Based Collaborative Filtering using K-Nearest Neighbors (`CineMatchKNN`).
- `src/eda.py`: Descriptive statistical calculations and distribution plots.
- `src/evaluation.py`: Stratified user-level train-test splitting, Precision@K and Recall@K computation.

---

### 2.1 Datasets & Schema Definition

The project leverages the benchmark MovieLens dataset:
1. **`movies.csv` (9,742 records):**
   - `movieId` (*integer*): Unique movie primary identifier.
   - `title` (*string*): Official movie title including release year (e.g., `"Matrix, The (1999)"`).
   - `genres` (*string*): Pipe-delimited list of genre tags (e.g., `"Action|Sci-Fi|Thriller"`).
2. **`ratings.csv` (100,836 records):**
   - `userId` (*integer*): Anonymized user identifier (610 unique users).
   - `movieId` (*integer*): Foreign key referencing `movies.csv`.
   - `rating` (*float*): Explicit rating score from $0.5$ to $5.0$ in half-star increments.
   - `timestamp` (*integer*): Unix epoch timestamp of submission.

---

### 2.2 Data Preprocessing & Sparsity Reduction Pipeline

#### A. Data Ingestion & Schema Verification (`load_raw_data`)
- Verifies file presence on disk.
- Ensures all mandatory columns exist: `movieId`, `title`, `genres` in `movies.csv` and `userId`, `movieId`, `rating` in `ratings.csv`.

#### B. Cleaning & Deduplication (`clean_and_merge_data`)
1. **Deduplication:** Removes duplicate `movieId` entries and duplicate `(userId, movieId)` interactions.
2. **Column Pruning:** Drops `timestamp` column as matrix collaborative filtering relies solely on user-item interaction scores.
3. **Null Handling & Rating Validation:** Rejects any records with null identifiers and asserts ratings fall within $[0.5, 5.0]$.
4. **Feature Engineering (Year Extraction):**
   - Applies regular expression `r"\((\d{4})\)"` on movie titles.
   - Parses the numeric release year into a distinct metadata column (e.g., `"Toy Story (1995)"` $\rightarrow$ `1995`).
5. **Metadata Aggregation:**
   - Precomputes total ratings count (`num_ratings`) and arithmetic average rating (`avg_rating`) per movie.

#### C. The Matrix Sparsity Problem & Noise Reduction (`filter_sparse_interactions`)
In recommendation systems, explicit feedback matrices are inherently sparse because users rate only a tiny subset of the global movie catalog.

**Mathematical Formula for Matrix Sparsity:**
$$\text{Sparsity} = \left( 1 - \frac{|\text{Total Interactions}|}{|\text{Unique Users}| \times |\text{Unique Movies}|} \right) \times 100\%$$

- **Raw Dataset Sparsity:**
  $$\text{Sparsity}_{\text{raw}} = \left( 1 - \frac{100,836}{610 \times 9,742} \right) \times 100\% \approx 98.30\%$$
- **Noise Filtering Strategy:**
  - Infrequently rated movies introduce high-variance noise (e.g., a movie with 1 rating of 5.0).
  - Users with too few ratings lack sufficient overlap to compute meaningful cosine angles.
  - Applying thresholds `min_movie_ratings = 10` and `min_user_ratings = 5` yields **2,269 qualified movies** and **610 users**.
- **Filtered Dataset Sparsity:**
  $$\text{Sparsity}_{\text{filtered}} = \left( 1 - \frac{79,636}{610 \times 2,269} \right) \times 100\% \approx 94.14\%$$
  *(Over 4.16% density gain, dramatically sharpening KNN vector representations).*

#### D. Item-User Pivot Matrix & CSR Compression (`create_pivot_matrix`)
- The filtered dataframe is pivoted into a 2D matrix:
  - **Rows:** Movie titles ($2,269$).
  - **Columns:** User IDs ($610$).
  - **Values:** Explicit ratings ($0.5$ to $5.0$). Unrated pairs are imputed with $0.0$.
- To ensure optimal memory usage and accelerated linear algebra during distance computations, the pivot matrix is compressed into **Compressed Sparse Row (`scipy.sparse.csr_matrix`)** format.

---

### 2.3 Item-Based Collaborative Filtering (KNN) Algorithm

#### Why Item-Based Collaborative Filtering?
1. **Taste Stability:** Item-to-item rating correlations remain consistent over time, whereas individual user preferences drift dynamically.
2. **Computational Scalability:** The item universe is more static than a continuously expanding user base, allowing vector representations to be queried with sub-millisecond latency.
3. **Transparent Explainability:** Recommendations are intuitive and interpretable (e.g., *"Users who rated Inception highly also rated Interstellar and The Dark Knight highly"*).

#### Mathematical Formulation

Each movie $i$ is modeled as a 610-dimensional vector in user rating space:
$$\vec{u} = [r_{u,1}, r_{u,2}, \dots, r_{u,610}]^T$$

**1. Cosine Similarity:**
Cosine similarity measures the cosine of the angle between two multi-dimensional movie vectors:
$$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \cos(\theta) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2} = \frac{\sum_{k=1}^{610} r_{u,k} r_{v,k}}{\sqrt{\sum_{k=1}^{610} r_{u,k}^2} \sqrt{\sum_{k=1}^{610} r_{v,k}^2}}$$

**2. Cosine Distance (Metric Loss):**
$$\text{Cosine Distance} = 1 - \text{Cosine Similarity}(\vec{u}, \vec{v})$$
- Distance bounds: $[0.0, 1.0]$.
- A distance of $0.0$ signifies identical user rating patterns; a distance of $1.0$ indicates orthogonal (completely unaligned) profiles.

**3. Nearest Neighbors Search (`CineMatchKNN`):**
- Uses Scikit-Learn's `NearestNeighbors(metric="cosine", algorithm="brute")`.
- When an input movie vector $\vec{x}$ is queried, the algorithm retrieves the top $K$ neighbors that minimize cosine distance:
  $$\arg\min_{j \ne i} \text{Cosine Distance}(\vec{x}_i, \vec{x}_j)$$
- **Similarity Percentage:**
  $$\text{Match Percentage} = \max\left(0, (1 - \text{distance})\right) \times 100\%$$

---

### 2.4 Title Normalization & Multi-Tier Search Engine (`find_movie`)

User queries often vary in spelling or formatting (e.g., typing `"The Matrix"` vs. `"Matrix, The (1999)"`). `CineMatchKNN.find_movie` implements an intelligent 3-tier matching engine:
1. **Tier 1 (Exact Match):** Direct case-insensitive string equality check.
2. **Tier 2 (Transposed Article Parsing):**
   - Detects leading articles: `"the "`, `"a "`, `"an "`.
   - Generates transposed variants (e.g., query `"The Matrix"` produces `"Matrix, The"` and `"Matrix"`).
   - Prioritizes prefixes and shortest string lengths.
3. **Tier 3 (Token-Based Overlap Scoring):**
   - Filters common stop words (`the`, `a`, `and`, `of`, `in`, `to`, `film`, `movie`).
   - Computes set intersection between query tokens and database title tokens.
   - Requires $\ge 50\%$ word overlap and ranks by $(\text{overlap count}, -\text{string length})$.

---

### 2.5 Dynamic Genre Filtering with Neighbor Expansion

When a user requests recommendations restricted to a specific genre (e.g., `"Animation"` or `"Sci-Fi"`):
- The model expands the candidate retrieval pool:
  $$N_{\text{fetch}} = \min(|M|, \max(4K + 1, 30))$$
- It evaluates the nearest neighbors in order of ascending cosine distance.
- It verifies each candidate's genre list from `metadata_lookup`.
- The first $K$ movies satisfying the genre criterion are returned, ensuring recommendations adhere strictly to user preferences without sacrificing collaborative filtering quality.

---

### 2.6 Exploratory Data Analysis (EDA) & Descriptive Statistics

Computed on the complete explicit interaction set ($N = 100,836$ ratings in `src/eda.py`):

| Statistical Metric | Empirical Value | Statistical Interpretation |
| :--- | :---: | :--- |
| **Count ($N$)** | $100,836$ | Complete interaction corpus size |
| **Mean ($\mu$)** | $3.50$ | Average rating across all users |
| **Median** | $3.50$ | 50th percentile (balanced central midpoint) |
| **Mode** | $4.00$ | Most frequently awarded rating |
| **Standard Deviation ($\sigma$)** | $1.04$ | Average dispersion around the mean |
| **Variance ($\sigma^2$)** | $1.09$ | Degree of spread in rating distributions |
| **Skewness** | $-0.64$ | **Negative (Left) Skew:** Asymmetry indicating users selectively rate movies they enjoy |
| **IQR ($Q_3 - Q_1$)** | $1.00$ | 50% of all ratings lie between 3.0 ($Q_1$) and 4.0 ($Q_3$) |
| **Valid Range** | $[0.5, 5.0]$ | Valid rating boundaries |

#### EDA Visualizations (`src/eda.py:save_all_plots`)
- **Rating Distribution Plot:** Histogram with overlaid Kernel Density Estimation (KDE) and vertical lines for Mean ($3.50$) and Median ($3.50$).
- **Top Movies Bar Chart:** Most reviewed movies in the dataset (led by *Forrest Gump*, *The Shawshank Redemption*, *Pulp Fiction*).
- **Genre Distribution Chart:** Breakdown of catalog titles across Drama, Comedy, Action, Thriller, Adventure, etc.
- **User Activity Distribution Plot:** Log-scale distribution demonstrating power-law (long-tail) behavior in user engagement.

---

### 2.7 Offline Model Evaluation (Precision@K & Recall@K)

To evaluate recommendation accuracy, `src/evaluation.py` performs offline benchmarking against real held-out user interactions.

#### A. Evaluation Protocol
1. **User Selection:** Active users with $\ge 15$ ratings.
2. **Stratified Split:** Partitions each user's ratings into **80% Training** and **20% Test** sets.
3. **Relevance Threshold:** A test interaction is defined as **Relevant (Ground Truth)** if:
   $$\text{Rating} \ge 3.5$$
4. **Seed Selection:** The evaluation queries the user's highest-rated movie in the training set and computes top $K$ nearest neighbors.

#### B. Mathematical Metric Definitions

$$\text{Precision@K} = \frac{|\text{Top } K \text{ Recommended Items} \cap \text{Relevant Test Items}|}{K}$$

$$\text{Recall@K} = \frac{|\text{Top } K \text{ Recommended Items} \cap \text{Relevant Test Items}|}{|\text{Total Relevant Test Items for User}|}$$

#### C. Benchmark Results for CineMatch KNN

Evaluated on 100+ active held-out test users:

| Metric | CineMatch KNN Performance |
| :--- | :---: |
| **Precision@5** | **6.89%** |
| **Recall@5** | **3.67%** |
| **Precision@10** | **5.27%** |
| **Recall@10** | **6.02%** |

#### Key Takeaway
- **The Precision-Recall Trade-off:** As $K$ increases from $5$ to $10$, **Recall increases** from $3.67\%$ to $6.02\%$ because more relevant movies are successfully captured in the larger recommendation set, while **Precision adjusts** from $6.89\%$ to $5.27\%$ because earlier ranks contain the highest cosine similarity scores.

---

## 3. Backend Architecture (Flask REST API)

The backend is built with Python and Flask (`server.py`), providing a lightweight, stateless REST API.

### 3.1 Application Lifecycle & In-Memory Pipeline State

When `server.py` starts:
1. `initialize_app_data()` executes during server boot.
2. Datasets `movies.csv` and `ratings.csv` are ingested and processed via `prepare_data_pipeline`.
3. The `CineMatchKNN` model is fitted on the CSR pivot matrix and stored in a global in-memory state dictionary (`STATE`).
4. Autocomplete catalogues and unique genre sets are precomputed for sub-5ms lookup times on search queries.
5. Cross-Origin Resource Sharing (CORS) headers are attached via `@app.after_request` to permit frontend calls.

```python
STATE = {
    "data": None,               # Processed DataFrames and CSR matrix
    "knn": None,                # Fitted CineMatchKNN instance
    "stats": None,              # Precomputed descriptive statistics
    "genres_list": [],          # Alphabetical list of unique genres
    "movies_list": [],          # Pre-sorted catalog of 2,269 movies
    "cached_evaluation": None   # Cached Precision@K / Recall@K results
}
```

---

### 3.2 REST API Endpoints Specification

#### 1. `GET /api/health`
- **Purpose:** Verifies backend status and provides matrix dimension telemetry.
- **Response:**
  ```json
  {
    "status": "online",
    "service": "CineMatch KNN Movie Recommendation API",
    "movies_count": 2269,
    "matrix_movies": 2269,
    "matrix_users": 610
  }
  ```

#### 2. `GET /api/movies`
- **Purpose:** Fast catalog lookup and autocomplete suggestions.
- **Query Parameters:**
  - `q` (*optional string*): Substring search for movie title.
  - `genre` (*optional string*): Filter movies belonging to genre.
  - `limit` (*optional int, default 100*): Number of results to return.
- **Response:**
  ```json
  {
    "total": 6,
    "genres": ["Action", "Adventure", "Animation", "Comedy", ...],
    "movies": [
      {
        "title": "Inception (2010)",
        "in_matrix": true,
        "year": 2010,
        "genres": "Action|Crime|Drama|Mystery|Sci-Fi|Thriller",
        "avg_rating": 4.07,
        "num_ratings": 143
      }
    ]
  }
  ```

#### 3. `GET /api/recommend`
- **Purpose:** Generates personalized Item-Based KNN recommendations for a given seed movie.
- **Query Parameters:**
  - `movie` (*required string*): Movie title query (e.g., `"Inception"` or `"Toy Story"`).
  - `top_k` (*optional int, default 6*): Number of recommendations to return.
  - `genre` (*optional string*): Restrict recommended items to a specific genre.
- **Response:**
  ```json
  {
    "status": "success",
    "searched_query": "Inception",
    "matched_movie": "Inception (2010)",
    "seed_details": {
      "title": "Inception (2010)",
      "genres": "Action|Crime|Drama|Mystery|Sci-Fi|Thriller",
      "year": 2010,
      "avg_rating": 4.07,
      "num_ratings": 143
    },
    "recommendations": [
      {
        "title": "Dark Knight, The (2008)",
        "similarity_score": 0.7279,
        "similarity_percentage": 72.79,
        "genres": "Action|Crime|Drama|IMAX",
        "year": 2008,
        "avg_rating": 4.24,
        "num_ratings": 149
      },
      {
        "title": "Inglourious Basterds (2009)",
        "similarity_score": 0.6461,
        "similarity_percentage": 64.61,
        "genres": "Action|Drama|War",
        "year": 2009,
        "avg_rating": 4.14,
        "num_ratings": 88
      }
    ]
  }
  ```

#### 4. `GET /api/stats`
- **Purpose:** Returns comprehensive descriptive statistics, rating frequency distribution table, top-rated movies, and genre breakdown for interactive visualization.

#### 5. `GET /api/evaluation`
- **Purpose:** Triggers or returns cached offline evaluation of the KNN model.

---

### 3.3 Error Handling & Edge Cases
- **Missing Query:** Calling `/api/recommend` without `?movie=` returns HTTP 400 Bad Request.
- **Unknown Title:** When a title is not found in the matrix, the API returns HTTP 200 with `status: "not_found"` and candidate suggestions if available.
- **Dynamic Genre Depletion:** If a genre filter is applied that eliminates immediate neighbors, the recommender dynamically expands neighbor retrieval depth up to $\min(|M|, 4K + 1, 30)$ to fulfill the request.

---

## 4. Frontend Architecture (React + Vite Web App)

The user-facing application is a single-page application built with **React 18** and **Vite** located in `frontend/`.

### 4.1 Tech Stack & UI Philosophy
- **Framework:** React 18 with Functional Components and React Hooks (`useState`, `useEffect`).
- **Icons:** `lucide-react` (clean SVG icons for search, stars, films, loading spinners).
- **Styling:** Custom CSS design system (`index.css` and `App.css`) featuring:
  - Clean light-mode aesthetic with slate backgrounds and indigo accents (`#4f46e5`).
  - Inter-based modern typography.
  - Interactive micro-animations (scale on hover, smooth transitions).
  - Subtle drop shadows (`box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1)`).

---

### 4.2 Client State & Interactive Components (`frontend/src/App.jsx`)

1. **State Hooks:**
   - `searchQuery`: Controlled input state for the search bar.
   - `suggestions`: Array of autocomplete options returned by `/api/movies?q=...`.
   - `selectedGenre` & `topK`: Filter controls for genre and result count (6, 9, or 12).
   - `recommendations`: Current result object containing seed metadata and recommendation cards.
   - `isLoading` & `errorMessage`: Async loading spinner and error notifications.

2. **Core Interactive Features:**
   - **Debounced Autocomplete:** Typing $\ge 2$ characters immediately queries `/api/movies?q=...` to display an autocomplete dropdown with movie ratings and vote counts.
   - **Quick-Pick Chips:** Preconfigured buttons for iconic movies (*Inception*, *Toy Story*, *The Matrix*, *The Dark Knight*, *Interstellar*, *Fight Club*) allowing single-click exploration.
   - **Seed Movie Spotlight:** When recommendations are loaded, a dedicated spotlight card showcases the seed movie's details (release year, rating, vote count, genres).
   - **Recommendation Grid Cards:**
     - Match badge displaying similarity percentage (`XX.XX% Match`).
     - Release year tag.
     - Star rating with total user votes.
     - Interactive genre badges.

---

### 4.3 Proxy Configuration & API Integration (`vite.config.js`)

Vite includes a built-in reverse proxy, preventing CORS issues in development:
```javascript
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true
      }
    }
  }
})
```
All frontend calls to `/api/*` are automatically forwarded to the Flask API at `http://127.0.0.1:5000`.

---

## 5. Complete End-to-End Data Flow

Let's trace what happens when a user types **"Inception"** and clicks Search:

1. **User Action:** The user types `"Inception"` and hits Enter or clicks the Find Similar button.
2. **Frontend Dispatch:** `App.jsx` invokes `fetchRecommendations("Inception", 6, "")`.
   - `isLoading` state becomes `true` (rendering the `Loader2` spinning icon).
3. **HTTP Request:** The browser sends `GET /api/recommend?movie=Inception&top_k=6` to Vite (`:5173`).
4. **Vite Reverse Proxy:** Vite forwards the request to Flask (`http://127.0.0.1:5000/api/recommend?movie=Inception&top_k=6`).
5. **Backend Processing (`server.py`):**
   - Retrieves `STATE["knn"]`.
   - Calls `knn.recommend("Inception", top_k=6)`.
6. **Title Resolution (`find_movie`):**
   - The string `"Inception"` is matched to `"Inception (2010)"` in the matrix index.
7. **Vector Neighbor Query (`NearestNeighbors`):**
   - Looks up the row index for `"Inception (2010)"` in `pivot_df`.
   - Extracts the 610-dimensional user rating vector $\vec{x}$.
   - Runs `model.kneighbors(\vec{x}, n_neighbors=25)`.
   - Computes Cosine Distance for each neighbor.
   - Excludes `"Inception (2010)"` (self-match).
8. **Metadata Enrichment:**
   - For each recommended movie, retrieves genres, release year, average rating, and vote counts from `metadata_lookup`.
   - Calculates `similarity_percentage = (1 - distance) * 100`.
9. **JSON Response:** Flask returns JSON payload containing seed details and recommendation items.
10. **Frontend Render:**
    - `isLoading` becomes `false`.
    - The spotlight banner renders `"Inception (2010)"` metadata.
    - The movie grid renders 6 cards (e.g., *The Dark Knight*, *Inglourious Basterds*, *Shutter Island*, *Fight Club*) with match percentages.

---

## 6. Automated Unit Testing & Verification

The project includes an automated test suite in `tests/test_recommender.py` utilizing Python's `unittest` framework.

### Test Coverage Summary (8 Test Cases)
1. **`test_schema_and_cleaning`:** Verifies required schema columns, absence of nulls, and asserts rating bounds ($0.5 \le r \le 5.0$).
2. **`test_sparsity_formula`:** Verifies sparsity calculation math (e.g., 10 users, 10 items, 20 ratings $\rightarrow 80.0\%$ sparsity).
3. **`test_pivot_matrix_dimensions`:** Asserts pivot matrix dimensions $> 0$ and no remaining NaN values.
4. **`test_knn_recommendations_valid_movie`:** Asserts querying `"Toy Story"` returns exactly $K$ items with similarity scores within $[0.0, 1.0]$.
5. **`test_knn_genre_filter`:** Verifies that requesting a genre filter (e.g., `"Animation"`) guarantees all recommendations contain that genre.
6. **`test_edge_case_unknown_movie`:** Asserts that querying a fictional movie returns `status: "not_found"` gracefully without crashing.
7. **`test_edge_case_empty_input`:** Asserts empty strings return `not_found` safely.
8. **`test_descriptive_statistics`:** Verifies statistical computations (`mean`, `median`, `variance`, `skewness`) and verifies mean rating $\approx 3.50$.

**Running the Test Suite:**
```bash
python -m unittest discover tests
```
*Expected Output:*
```text
Ran 8 tests in 0.586s
OK
```

---

## 7. Execution & Run Guide

### Prerequisites
- **Python 3.9+**
- **Node.js 18+** & npm

---

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Web Application

**Terminal 1: Start Backend (Flask Web Service)**
```bash
python movie_recommender.py
```
*(Server listens at `http://localhost:5000`)*

**Terminal 2: Start Frontend (React UI)**
```bash
cd frontend
npm install
npm run dev
```
*(Open your browser at `http://localhost:5173`)*


---

## 8. Viva & Technical Interview Guide (FAQ)

### Q1: What is Collaborative Filtering?
> **Answer:** Collaborative Filtering is a recommendation paradigm that produces suggestions based on historical user interaction patterns rather than explicit item features. Its core premise is that users who agreed on items in the past will agree on items in the future.

### Q2: Why did you choose Item-Based over User-Based Collaborative Filtering?
> **Answer:**
> 1. **Stability:** Item-item relationships are far more stable over time than user preference profiles, which fluctuate frequently.
> 2. **Scalability:** In typical commercial applications, the number of users greatly exceeds the number of items. Precomputing an item-item matrix is more space- and time-efficient.
> 3. **Reduced Drift:** It prevents user taste drift from degrading recommendations.

### Q3: Why is Cosine Similarity preferred over Euclidean Distance?
> **Answer:** Euclidean distance measures absolute spatial distance, which is severely distorted by user rating scale bias (e.g., a "critical" user who rates good movies 3 stars vs. an "enthusiastic" user who rates good movies 5 stars). Cosine similarity measures the angle between vectors, capturing relative preference patterns regardless of scale.

### Q4: What is the Sparsity Problem and how did you resolve it?
> **Answer:** Sparsity occurs because any individual user rates only a tiny fraction of the total catalog. In our raw MovieLens dataset, sparsity was **98.30%**. We addressed this by filtering out movies with $<10$ ratings and users with $<5$ ratings. This reduced sparsity to **94.14%**, eliminating noise and cold-start artifacts.

### Q5: What does the negative skewness ($-0.64$) in the rating distribution signify?
> **Answer:** A negative skew (tail extending to the left) means that the majority of ratings are concentrated toward the higher end (Mode = 4.0, Mean = 3.50). In recommendation theory, this reflects self-selection bias: users voluntarily choose to watch and rate movies they anticipate enjoying, rather than picking movies at random.

### Q6: Why does Recall increase while Precision decreases when $K$ goes from 5 to 10?
> **Answer:** This represents the classic Information Retrieval trade-off:
> - As $K$ expands, the model casts a wider net, capturing a higher proportion of all relevant test items (**Recall increases** from 3.67% to 6.02%).
> - However, the items appearing at positions 6 through 10 have slightly lower cosine similarity than the top 5, resulting in a slightly lower proportion of true hits per slot (**Precision adjusts** from 6.89% to 5.27%).
