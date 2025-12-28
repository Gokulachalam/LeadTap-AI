import json
import re
import logging
from pathlib import Path

import requests
import ollama
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


# Basic app config
QDRANT_URL = "http://localhost:6333"
COLLECTION = "properties"
VECTOR_SIZE = 768
DATA_PATH = "data/sample-dataset.json"
STATIC_DIR = Path(__file__).parent / "static"


app = FastAPI(title="Semantic Property Search")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Needed for browser requests from HTML
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Generate vector embedding using Ollama
def embed(text: str) -> list:
    try:
        res = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )
        return res["embedding"]
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise


# Supported amenities
AMENITIES = [
    "parking",
    "balcony",
    "pet_friendly",
    "garden",
    "public_transit",
    "gym",
    "pool",
    "terrace",
]


# Extract filters from user query
def parse_query(query: str) -> dict:
    q = query.lower()
    parsed = {}

    price = re.search(r"(under|below|less than)\s*(\d+)", q)
    if price:
        parsed["max_price"] = int(price.group(2))

    beds = re.search(r"(\d+)\s*(bedroom|bhk|rk)", q)
    if beds:
        parsed["min_bedrooms"] = int(beds.group(1))

    parsed["amenities"] = [a for a in AMENITIES if a in q]

    if "school" in q:
        parsed["prefer_school"] = True

    loc = re.search(
        r"(near|in|at)\s+([a-z\s]+?)(?:\s+under|\s+below|\s+\d|$)",
        q
    )
    if loc:
        parsed["location"] = loc.group(2).strip()

    return parsed


# Load dataset into Qdrant once
def ingest_data():
    logger.info("Starting ingestion")

    collections = requests.get(
        f"{QDRANT_URL}/collections"
    ).json()["result"]["collections"]

    names = [c["name"] for c in collections]

    if COLLECTION not in names:
        requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION}",
            json={
                "vectors": {
                    "size": VECTOR_SIZE,
                    "distance": "Cosine"
                }
            }
        )

    count = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/count",
        json={"exact": True}
    ).json()["result"]["count"]

    if count > 0:
        logger.info("Data already exists")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        properties = json.load(f)

    points = []

    for idx, prop in enumerate(properties, start=1):
        text = (
            f"{prop['title']}\n"
            f"{prop['description']}\n"
            f"Neighborhood: {prop['neighborhood']}\n"
            f"Amenities: {', '.join(prop['amenities'])}\n"
            f"Nearby: {json.dumps(prop['nearby_places'])}"
        )

        vector = embed(text)

        points.append({
            "id": idx,
            "vector": vector,
            "payload": {
                "property_id": prop["id"],
                "title": prop["title"],
                "price": prop["price"],
                "bedrooms": prop["bedrooms"],
                "neighborhood": prop["neighborhood"],
                "amenities": prop["amenities"],
                "nearby_places": prop["nearby_places"]
            }
        })

    requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        json={"points": points}
    )

    logger.info("Ingestion completed")


# Vector search + scoring
def search_properties(query: str, parsed: dict):
    qvec = embed(query)

    res = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        json={
            "vector": qvec,
            "limit": 50,
            "with_payload": True
        }
    ).json()["result"]

    ranked = []

    for r in res:
        p = r["payload"]
        score = r["score"]
        reasons = [f"semantic={round(score, 2)}"]

        # Hard filters
        if "max_price" in parsed and p["price"] > parsed["max_price"]:
            continue
        if "min_bedrooms" in parsed and p["bedrooms"] < parsed["min_bedrooms"]:
            continue

        # Soft boosts
        if "location" in parsed and parsed["location"] in p["neighborhood"].lower():
            score += 0.15
            reasons.append("location")

        matched = set(parsed.get("amenities", [])) & set(p["amenities"])
        if matched:
            score += 0.05 * len(matched)
            reasons.append(f"amenities={list(matched)}")

        if parsed.get("prefer_school"):
            for place in p.get("nearby_places", []):
                if place.get("type") == "school" and place.get("distance_m", 9999) <= 300:
                    score += 0.1
                    reasons.append("school_nearby")
                    break

        ranked.append({
            "id": p["property_id"],
            "title": p["title"],
            "score": round(score, 3),
            "reasons": reasons,
            "metadata": p
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:5]


@app.get("/")
async def serve_ui():
    return HTMLResponse(
        open(STATIC_DIR / "index.html", encoding="utf-8").read()
    )


@app.post("/search")
async def search_api(req: Request):
    try:
        data = await req.json()
        query = data.get("query", "")

        parsed = parse_query(query)
        results = search_properties(query, parsed)

        return JSONResponse({
            "parsed_query": parsed,
            "results": results
        })
    except Exception as e:
        logger.error(e)
        return JSONResponse({"error": str(e)}, status_code=500)


# Load data when app starts
@app.on_event("startup")
async def on_startup():
    ingest_data()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
