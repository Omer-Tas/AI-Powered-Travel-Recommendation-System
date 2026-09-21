# WanderSync: AI-Powered Travel Recommendation Engine 🌍✈️

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![NLP](https://img.shields.io/badge/NLP-Transformers-orange)
![Machine Learning](https://img.shields.io/badge/Machine--Learning-Scikit--Learn-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📌 Overview

Traditional travel platforms often rely on static, rule-based filtering such as selecting a destination, date, or predefined category. These approaches may fail to capture the nuances of a user's actual travel preferences.

**WanderSync** is an end-to-end, AI-powered travel recommendation engine designed to transform natural-language travel requests into personalized destination, attraction, accommodation, and restaurant recommendations.

Instead of relying primarily on rigid `if-else` rules, WanderSync combines:

* Natural Language Processing (NLP)
* Text embeddings and vector representations
* Unsupervised learning
* Clustering
* Cosine similarity
* Recommendation algorithms
* Sentiment analysis
* User preferences and budget constraints

The system maps users and destinations into a multidimensional representation space and uses these representations to generate personalized travel recommendations.

---

## 🚀 Key Features

### 1. Semantic Destination Matching

Processes free-text travel requests using NLP techniques and identifies destinations that best match the user's intent.

Example:

> "I want a quiet, budget-friendly beach holiday with historical ruins nearby."

The system extracts the semantic characteristics of the request and compares them against destination profiles.

### 2. Top-10 Attraction Recommendation

After selecting a destination, WanderSync recommends the most relevant attractions using historical ratings and review information.

The recommendation process uses a weighted-rating approach to reduce the impact of attractions with very few reviews.

### 3. Sentiment-Aware Accommodation & Dining

Hotels and restaurants are evaluated using customer reviews and NLP-based sentiment analysis.

Instead of relying solely on average ratings, the system analyzes positive, neutral, and negative feedback to estimate overall customer satisfaction.

### 4. End-to-End Personalization

The recommendation pipeline considers multiple user-specific factors, including:

* Travel preferences
* Travel style
* Budget
* Destination characteristics
* Attraction preferences
* Accommodation preferences
* Restaurant preferences
* Review sentiment

---

## 🧠 System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    USER FREE-TEXT INPUT                     │
│                                                             │
│ "I want a relaxing and affordable beach holiday..."        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1 — INTENT DETECTION & TEXT VECTORIZATION             │
│                                                             │
│ • Text preprocessing                                        │
│ • Tokenization / Lemmatization                              │
│ • TF-IDF / Embeddings                                        │
│ • Intent representation                                     │
│                                                             │
│ Output: User Intent Vector                                  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2 — DESTINATION MATCHING                              │
│                                                             │
│ • Destination profile vectorization                         │
│ • K-Means clustering                                        │
│ • Vector-space mapping                                      │
│ • Cosine similarity                                         │
│                                                             │
│ Output: Selected Destination / Country                      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3 — POI & AMENITIES RECOMMENDATION                    │
│                                                             │
│ ┌──────────────────────┐   ┌─────────────────────────────┐ │
│ │ Attractions          │   │ Hotels & Restaurants        │ │
│ │                      │   │                             │ │
│ │ Collaborative /      │   │ NLP Sentiment Analysis      │ │
│ │ Weighted Rating      │   │ + User Preference Matching  │ │
│ └──────────────────────┘   └─────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              FINAL PERSONALIZED ITINERARY                   │
│                                                             │
│ Destination → Attractions → Hotels → Restaurants            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📐 Technical Implementation

### 1. Intent Detection & Text Vectorization

The user's free-text request is transformed into a numerical representation using NLP techniques and embedding models.

For example:

> "I want a quiet, budget-friendly beach holiday with historical ruins nearby."

can be transformed into a dense vector representation.

Let $U$ represent the user's input document. The embedding function $E(U)$ maps the input into an $n$-dimensional vector space:

$$
\vec{v}_u = E(U) \in \mathbb{R}^n
$$

The resulting vector captures semantic characteristics of the user's travel request.

---

### 2. Destination Clustering & Similarity Matching

Destination profiles are represented as vectors containing characteristics such as:

* Climate
* Travel style
* Historical significance
* Cultural characteristics
* Budget level
* Beach / nature availability
* Culinary characteristics

K-Means clustering can then be used to group destinations with similar characteristics.

The K-Means objective function minimizes the within-cluster sum of squared distances:

$$
J = \sum_{i=1}^{k} \sum_{\vec{x} \in S_i}
\|\vec{x} - \vec{\mu}_i\|^2
$$

where:

* $S_i$ = cluster $i$
* $\vec{x}$ = destination vector
* $\vec{\mu}_i$ = centroid of cluster $i$
* $k$ = number of clusters

After generating the user vector, WanderSync calculates the cosine similarity between the user representation and destination representations.

$$
\text{Similarity}(u,d)
=
\cos(\theta)
=
\frac{\vec{v}_u \cdot \vec{v}_d}
{\|\vec{v}_u\| \times \|\vec{v}_d\|}
$$

Higher similarity indicates stronger alignment between the user's travel intent and the destination profile.

---

### 3. Attraction Recommendation

After selecting a destination, WanderSync generates a ranked list of attractions.

A weighted-rating approach is used to prevent attractions with a small number of reviews from dominating the ranking.

The weighted score is calculated as:

$$
W =
\left(
\frac{v}{v+m} \cdot R
\right)
+
\left(
\frac{m}{v+m} \cdot C
\right)
$$

where:

| Variable | Description                          |
| -------- | ------------------------------------ |
| $v$      | Number of reviews for the attraction |
| $m$      | Minimum number of reviews required   |
| $R$      | Average rating of the attraction     |
| $C$      | Mean rating across the dataset       |
| $W$      | Final weighted rating                |

The resulting scores are used to generate the **Top-10 attraction recommendations**.

---

### 4. NLP Sentiment Analysis for Hotels & Restaurants

Average star ratings alone may not fully represent customer satisfaction.

WanderSync therefore analyzes customer reviews using NLP-based sentiment analysis.

Reviews are classified into sentiment categories:

* Positive
* Neutral
* Negative

A sentiment-based satisfaction score can then be calculated as:

$$
TSS =
\frac{
\sum P_{score} -
\sum N_{score}
}{
\text{Total Reviews}
}
$$

where:

* $P_{score}$ = positive sentiment contribution
* $N_{score}$ = negative sentiment contribution
* $TSS$ = True Satisfaction Score

The resulting score can be combined with the user's intent and preferences to identify relevant hotels and restaurants.

---

## 🔄 Recommendation Pipeline

The complete recommendation workflow can be summarized as:

```text
User Free-Text Request
        │
        ▼
Text Preprocessing
        │
        ▼
NLP / Embedding Generation
        │
        ▼
User Intent Vector
        │
        ▼
Destination Vector Matching
        │
        ├── K-Means Clustering
        │
        └── Cosine Similarity
        │
        ▼
Selected Destination
        │
        ├───────────────┐
        ▼               ▼
Attractions       Hotels / Restaurants
        │               │
        ▼               ▼
Weighted Rating    Sentiment Analysis
        │               │
        └───────┬───────┘
                ▼
      Personalized Results
                │
                ▼
       Recommended Itinerary
```

---

## 🛠️ Tech Stack

### Programming

* **Python 3.9+**

### NLP & Deep Learning

* HuggingFace Transformers
* NLTK
* spaCy

### Machine Learning

* Scikit-Learn
* K-Means Clustering
* TF-IDF
* Cosine Similarity

### Data Processing

* Pandas
* NumPy
* DuckDB

### Backend & API

* FastAPI
* Uvicorn

### Frontend / Demo

* Streamlit

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/WanderSync.git
cd WanderSync
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the API

Start the FastAPI application using Uvicorn:

```bash
uvicorn app.main:app --reload
```

Once the server is running, the API will be available locally.

FastAPI's interactive API documentation can typically be accessed through:

```text
http://127.0.0.1:8000/docs
```

---

## 🧪 Example Request

A recommendation request can be sent to the API using a JSON payload such as:

```json
{
  "user_input": "I am looking for a relaxing, warm climate with rich culinary culture and affordable lodging."
}
```

The system processes the request through the recommendation pipeline and returns destination and travel-related recommendations.

---

## 📊 Example Recommendation Flow

### User Input

```text
I want a relaxing, warm destination with beautiful beaches,
good food, historical places and affordable accommodation.
```

### NLP Processing

```text
Travel Style:
Relaxing

Climate:
Warm

Activities:
Beach + Historical

Food:
High Preference

Budget:
Affordable
```

### Destination Matching

```text
User Intent Vector
        │
        ▼
Destination Vector Space
        │
        ▼
Similarity Calculation
        │
        ▼
Best-Matching Destinations
```

### Final Recommendation

```text
Destination
    │
    ├── Top Attractions
    │
    ├── Recommended Hotels
    │
    └── Recommended Restaurants
```

---

## 🎯 Design Goals

WanderSync is designed around four main objectives:

### Personalization

Move beyond static filters by interpreting the user's natural-language preferences.

### Semantic Understanding

Represent both user intent and destinations within a common vector space.

### Data-Driven Recommendations

Use historical ratings, reviews, clustering, and sentiment analysis to support recommendation decisions.

### End-to-End Architecture

Combine NLP, machine learning, recommendation algorithms, data processing, and API services into a single recommendation pipeline.

---

## 🔮 Future Improvements

Potential extensions include:

* Transformer-based fine-tuned intent classification
* Context-aware conversational recommendations
* Real-time flight and hotel price integration
* Multi-destination itinerary optimization
* Reinforcement learning for recommendation ranking
* Graph-based travel recommendation
* Time-aware recommendation models
* Personalized travel budget optimization
* Weather-aware destination selection
* Real-time review and sentiment monitoring
* Large Language Model (LLM) integration
* Retrieval-Augmented Generation (RAG)
* Explainable recommendation generation

---

## 📌 Project Status

**Active Development**

WanderSync is an experimental AI-powered travel recommendation project combining NLP, machine learning, recommendation systems, and sentiment analysis.

---

## 👨‍💻 Author

**Ömer Taş**

Computer Engineering
AI • Machine Learning • Data Science • Quantitative Systems
