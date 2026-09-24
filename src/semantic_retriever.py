from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"

# Minimum similarity score for evidence to be considered relevant
DEFAULT_MIN_SCORE = 0.50


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def prepare_records(records):
    """
    Keep only expert statements.
    Interviewer statements are excluded from retrieval.
    """
    expert_records = []

    for record in records:
        if record["speaker"].lower() != "interviewer":
            expert_records.append(record)

    return expert_records


def build_embeddings(records, model):
    """
    Create embeddings for all expert statements.
    """
    expert_records = prepare_records(records)

    texts = [record["text"] for record in expert_records]

    embeddings = model.encode(
        texts,
        convert_to_tensor=True
    )

    return expert_records, embeddings


def semantic_search(
    question,
    records,
    model,
    top_k=5,
    min_score=DEFAULT_MIN_SCORE
):
    """
    Retrieve the most semantically relevant expert statements.

    Only statements with a similarity score greater than or equal
    to min_score are returned.
    """

    expert_records, embeddings = build_embeddings(records, model)

    # If there are no expert statements
    if not expert_records:
        return []

    question_embedding = model.encode(
        question,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        question_embedding,
        embeddings
    )[0]

    # Sort from highest similarity to lowest
    sorted_indices = scores.argsort(descending=True)

    results = []

    for index in sorted_indices:

        index = int(index)
        score = float(scores[index])

        # Stop once scores fall below the threshold
        if score < min_score:
            break

        results.append({
            "timestamp": expert_records[index]["timestamp"],
            "speaker": expert_records[index]["speaker"],
            "text": expert_records[index]["text"],
            "score": score
        })

        # Keep only the requested number of results
        if len(results) >= top_k:
            break

    return results
