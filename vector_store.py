"""
vector_store.py - everything that talks to the Qdrant vector database.

How versions work (alias-based updating):
    Real collections are named   rag_docs_v20261107_183000, rag_docs_v20261107_191500, ...
    The ALIAS  "rag_docs"  is a nickname that points at ONE of them (the live one).
    Searches always use the alias. To update, we fill a brand-new collection, then move
    the alias to it in one atomic step. Searches never see a half-built collection, and
    the previous version is kept so you can roll back.

Look inside the database from the command line:
    python vector_store.py info                         aliases, versions, point counts
    python vector_store.py show [N]                     first N chunks with all their metadata
    python vector_store.py filter --source X.pdf --page 3   find chunks by metadata only
    python vector_store.py rollback                     point the alias at the previous version
    python vector_store.py use <collection>             point the alias at any version
    python vector_store.py cleanup                      delete old versions (keeps live + 1)

Or open the dashboard: http://localhost:6333/dashboard
"""
import sys
from datetime import datetime

from qdrant_client import QdrantClient, models

QDRANT_URL = "http://localhost:6333"
COLLECTION_ALIAS = "rag_docs"                  # the name searches use
KEEP_VERSIONS = 2                              # live version + 1 older one for rollback


def connect():
    # check_compatibility=False: the newest client for Python 3.9 is a bit older than the
    # server. The features used here (collections, aliases, payloads) work on both.
    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    try:
        client.get_collections()
    except Exception:
        sys.exit(f"Can't reach Qdrant at {QDRANT_URL}. Start it first: qdrant\\start_qdrant.bat")
    return client


# %% aliases and versions
def live_collection(client):
    """The real collection the alias points at right now (None if there's no alias yet)."""
    for a in client.get_aliases().aliases:
        if a.alias_name == COLLECTION_ALIAS:
            return a.collection_name
    return None


def list_versions(client):
    """All versions, oldest first (the timestamp in the name sorts correctly)."""
    prefix = COLLECTION_ALIAS + "_v"
    return sorted(c.name for c in client.get_collections().collections if c.name.startswith(prefix))


def switch_alias(client, collection):
    """Point the alias at `collection`. Delete + create are sent as ONE atomic request."""
    actions = []
    if live_collection(client):
        actions.append(models.DeleteAliasOperation(
            delete_alias=models.DeleteAlias(alias_name=COLLECTION_ALIAS)))
    actions.append(models.CreateAliasOperation(
        create_alias=models.CreateAlias(collection_name=collection, alias_name=COLLECTION_ALIAS)))
    client.update_collection_aliases(change_aliases_operations=actions)


def delete_old_versions(client, keep=KEEP_VERSIONS):
    """Delete old versions, keeping the live one plus the newest (keep - 1) others."""
    live = live_collection(client)
    others = [n for n in list_versions(client) if n != live]
    doomed = others[:max(0, len(others) - (keep - 1))]
    for name in doomed:
        client.delete_collection(name)
    return doomed


# %% writing and reading
def create_version(client, chunks, vectors):
    """Fill a NEW collection with the chunks, then switch the alias to it."""
    name = f"{COLLECTION_ALIAS}_v{datetime.now():%Y%m%d_%H%M%S}"
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=32, ef_construct=200),  # same as the FAISS version
    )
    # Payload indexes make metadata filters (one file, one page) fast.
    client.create_payload_index(name, "source", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "page", models.PayloadSchemaType.INTEGER)

    points = [models.PointStruct(id=c["chunk_id"], vector=v.tolist(), payload={**c, "version": name})
              for c, v in zip(chunks, vectors)]
    for start in range(0, len(points), 256):
        client.upsert(name, points=points[start:start + 256], wait=True)

    switch_alias(client, name)                 # only now do searches see the new data
    return name


def load_chunks(client, collection=COLLECTION_ALIAS):
    """Read every chunk's text + metadata (payload) back from Qdrant, ordered by chunk_id."""
    chunks, offset = [], None
    while True:
        points, offset = client.scroll(collection, limit=256, offset=offset,
                                       with_payload=True, with_vectors=False)
        chunks += [p.payload for p in points]
        if offset is None:
            break
    return sorted(chunks, key=lambda c: c["chunk_id"])


def search(client, query_vector, k=20, hnsw_ef=64):
    """Nearest chunks by meaning. hnsw_ef = search quality (like efSearch in FAISS)."""
    hits = client.query_points(
        COLLECTION_ALIAS, query=query_vector, limit=k,
        search_params=models.SearchParams(hnsw_ef=hnsw_ef), with_payload=False,
    ).points
    return [int(h.id) for h in hits]


# %% command-line tool
def _short(text, n=90):
    return text if len(text) <= n else text[:n] + "..."


def cmd_info(client):
    live = live_collection(client)
    print(f"Alias '{COLLECTION_ALIAS}' -> {live or '(not set)'}\n")
    for name in list_versions(client):
        info = client.get_collection(name)
        sources = {c["source"] for c in load_chunks(client, name)}
        mark = "  <- live" if name == live else ""
        print(f"  {name}  {info.points_count} chunks  {sorted(sources)}{mark}")


def cmd_show(client, n=5):
    points, _ = client.scroll(COLLECTION_ALIAS, limit=n, with_payload=True, with_vectors=True)
    for p in sorted(points, key=lambda p: p.id):
        vec = p.vector
        print(f"--- point id {p.id}")
        for key, value in p.payload.items():
            print(f"  {key:12} {_short(str(value))}")
        print(f"  {'vector':12} [{', '.join(f'{x:.3f}' for x in vec[:4])}, ...]  ({len(vec)} numbers)")


def cmd_filter(client, source=None, page=None):
    must = []
    if source:
        must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
    if page is not None:
        must.append(models.FieldCondition(key="page", match=models.MatchValue(value=page)))
    points, _ = client.scroll(COLLECTION_ALIAS, scroll_filter=models.Filter(must=must),
                              limit=50, with_payload=True)
    print(f"{len(points)} chunk(s) match")
    for p in sorted(points, key=lambda p: p.id):
        print(f"  id {p.id:4}  {p.payload['source']} p.{p.payload['page']}  {_short(p.payload['text'], 70)}")


def cmd_rollback(client):
    versions, live = list_versions(client), live_collection(client)
    if live not in versions or versions.index(live) == 0:
        sys.exit("No older version to roll back to.")
    previous = versions[versions.index(live) - 1]
    switch_alias(client, previous)
    print(f"Alias '{COLLECTION_ALIAS}': {live} -> {previous}")


def main(argv):
    client = connect()
    cmd = argv[0] if argv else "info"
    if cmd == "info":
        cmd_info(client)
    elif cmd == "show":
        cmd_show(client, int(argv[1]) if len(argv) > 1 else 5)
    elif cmd == "filter":
        opts = dict(zip(argv[1::2], argv[2::2]))
        page = opts.get("--page")
        cmd_filter(client, opts.get("--source"), int(page) if page else None)
    elif cmd == "rollback":
        cmd_rollback(client)
    elif cmd == "use" and len(argv) == 2:
        switch_alias(client, argv[1])
        print(f"Alias '{COLLECTION_ALIAS}' -> {argv[1]}")
    elif cmd == "cleanup":
        print("Deleted:", delete_old_versions(client) or "nothing")
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
