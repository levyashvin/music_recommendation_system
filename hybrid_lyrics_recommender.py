import argparse, os, re, sys
import numpy as np
import pandas as pd

def clean_lyrics(text):
    if not isinstance(text, str):
        return ""
    x = re.sub(r"\[.*?\]", " ", text)  # remove [Chorus], [Verse], etc.
    x = x.replace("\n", " ").lower()
    x = re.sub(r"[^a-z0-9' ]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def chunk_words(text, max_words=220):
    ws = text.split()
    if not ws:
        return []
    return [" ".join(ws[i:i+max_words]) for i in range(0, len(ws), max_words)]

def load_df(csv_path, id_col, title_col, artist_col, lyrics_col, limit_rows=None):
    usecols = [id_col, title_col, artist_col, lyrics_col]
    df = pd.read_csv(csv_path, nrows=limit_rows, usecols=usecols)
    for c in [id_col, title_col, artist_col, lyrics_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
        df[c] = df[c].astype(str)
    df[lyrics_col] = df[lyrics_col].apply(clean_lyrics)
    df = df[df[lyrics_col].str.len() > 0].drop_duplicates(
        subset=[title_col, artist_col, lyrics_col]
    ).reset_index(drop=True)
    return df

def build(csv_path, out_dir, id_col, title_col, artist_col, lyrics_col,
          model_name="sentence-transformers/all-MiniLM-L6-v2",
          ngram_low=1, ngram_high=1, max_df=0.6, min_df=10, max_features=100_000,
          batch_size=64, max_words=350, device="auto", tfidf="on", songs_per_batch=1024,
          # HNSW params (if used)
          ef_construction=200, M=64, ef=256,
          # Annoy params (if used)
          index_backend="annoy", annoy_trees=50,
          limit_rows=None):
    os.makedirs(out_dir, exist_ok=True)
    df = load_df(csv_path, id_col, title_col, artist_col, lyrics_col, limit_rows=limit_rows)

    # TF-IDF (optional, can be memory-heavy)
    X = None
    if str(tfidf).lower() in ("on", "true", "1", "yes"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import save_npz
        import joblib

        vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(ngram_low, ngram_high),
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            strip_accents="unicode",
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        X = vec.fit_transform(df[lyrics_col])
        joblib.dump(vec, os.path.join(out_dir, "tfidf_vectorizer.joblib"))
        save_npz(os.path.join(out_dir, "tfidf_matrix.npz"), X)

    # Embeddings (chunk long lyrics, mean-pool), then L2-normalize final vectors
    from sentence_transformers import SentenceTransformer
    try:
        import torch  # type: ignore
        if device in ("cpu", "cuda"):
            dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else ("cpu" if device == "cpu" else "cpu")
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    model = SentenceTransformer(model_name, device=dev)
    # Streaming embedding build to keep RAM low
    n_docs = len(df)
    emb_path = os.path.join(out_dir, "embeddings.npy")
    emb_mm = None
    dim = None

    def encode_texts(texts):
        return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)

    lyrics_list = df[lyrics_col].tolist()
    for start in range(0, n_docs, int(songs_per_batch)):
        end = min(n_docs, start + int(songs_per_batch))
        batch_texts = []
        ptrs = []
        for t in lyrics_list[start:end]:
            ch = chunk_words(t, max_words=max_words) or [t]
            s = len(batch_texts)
            batch_texts.extend(ch)
            e = len(batch_texts)
            ptrs.append((s, e))
        if not batch_texts:
            continue
        chunk_emb = encode_texts(batch_texts)
        if dim is None:
            dim = int(chunk_emb.shape[1])
            from numpy.lib.format import open_memmap
            emb_mm = open_memmap(emb_path, mode='w+', dtype=np.float32, shape=(n_docs, dim))
        for i, (s, e) in enumerate(ptrs):
            vec = chunk_emb[s:e].mean(axis=0)
            nrm = np.linalg.norm(vec)
            if nrm > 1e-12:
                vec = (vec / nrm).astype(np.float32)
            else:
                vec = vec.astype(np.float32)
            emb_mm[start + i] = vec
        del batch_texts, ptrs, chunk_emb
    if emb_mm is not None:
        try:
            emb_mm.flush()
        except Exception:
            pass
        del emb_mm

    # Build ANN index depending on backend
    be = (index_backend or "annoy").lower()
    if be == "hnswlib":
        try:
            import hnswlib  # type: ignore
        except Exception as e:
            raise RuntimeError("hnswlib requested but not available. Try --index-backend annoy or flat.") from e
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=emb.shape[0], ef_construction=ef_construction, M=M)
        index.add_items(emb, ids=np.arange(emb.shape[0], dtype=np.int64))
        index.set_ef(ef)
        index.save_index(os.path.join(out_dir, "embeddings_hnsw.bin"))
    elif be == "annoy":
        try:
            from annoy import AnnoyIndex  # type: ignore
        except Exception as e:
            raise RuntimeError("annoy requested but not available. Install via 'pip install annoy'.") from e
        ann = AnnoyIndex(dim, "angular")
        for i in range(emb.shape[0]):
            ann.add_item(i, emb[i].tolist())
        ann.build(int(annoy_trees))
        ann.save(os.path.join(out_dir, "embeddings_annoy.ann"))
    elif be == "flat":
        # No index file needed; we will use matrix multiplications at query time
        pass
    else:
        raise ValueError(f"Unsupported index backend: {index_backend}. Use one of: annoy, flat, hnswlib.")

    # Meta + lookup
    meta = df[[id_col, title_col, artist_col]].copy()
    meta.to_csv(os.path.join(out_dir, "meta.csv"), index=False)
    lk = pd.DataFrame({
        "row": np.arange(len(df), dtype=np.int32),
        "title_lc": df[title_col].str.lower(),
        "artist_lc": df[artist_col].str.lower(),
    })
    lk.to_csv(os.path.join(out_dir, "lookup.csv"), index=False)

    tfidf_info = f"TF-IDF={X.shape}" if X is not None else "TF-IDF=skipped"
    # Read shape from saved embeddings
    emb_shape = np.load(emb_path, mmap_mode='r').shape
    print(f"Built: {tfidf_info}, Embeddings={emb_shape}, out={out_dir}")

def load_runtime(out_dir, model_name=None, set_ef=256, index_backend="annoy", device="auto"):
    import joblib
    from scipy.sparse import load_npz

    vec = None
    X = None
    vec_path = os.path.join(out_dir, "tfidf_vectorizer.joblib")
    X_path = os.path.join(out_dir, "tfidf_matrix.npz")
    if os.path.exists(vec_path) and os.path.exists(X_path):
        vec = joblib.load(vec_path)
        X = load_npz(X_path)
    emb = np.load(os.path.join(out_dir, "embeddings.npy"), mmap_mode='r')
    meta = pd.read_csv(os.path.join(out_dir, "meta.csv"))
    lk = pd.read_csv(os.path.join(out_dir, "lookup.csv"))

    dim = emb.shape[1]
    be = (index_backend or "annoy").lower()
    index = None
    if be == "hnswlib":
        import hnswlib  # type: ignore
        index = hnswlib.Index(space="cosine", dim=dim)
        index.load_index(os.path.join(out_dir, "embeddings_hnsw.bin"))
        index.set_ef(set_ef)
    elif be == "annoy":
        from annoy import AnnoyIndex  # type: ignore
        index = AnnoyIndex(dim, "angular")
        index.load(os.path.join(out_dir, "embeddings_annoy.ann"))
    elif be == "flat":
        index = None
    else:
        raise ValueError(f"Unsupported index backend: {index_backend}. Use one of: annoy, flat, hnswlib.")

    model = None
    if model_name:
        from sentence_transformers import SentenceTransformer
        try:
            import torch  # type: ignore
            if device in ("cpu", "cuda"):
                dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else ("cpu" if device == "cpu" else "cpu")
            else:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"
        model = SentenceTransformer(model_name, device=dev)
    return vec, X, emb, meta, lk, index, model

def find_song_row(lk, title, artist=None):
    t = str(title).lower()
    mask = lk["title_lc"] == t
    if artist:
        mask &= lk["artist_lc"] == str(artist).lower()
    if not mask.any():
        mask = lk["title_lc"].str.contains(re.escape(t), na=False)
        if artist:
            mask &= lk["artist_lc"].str.contains(re.escape(str(artist).lower()), na=False)
    if not mask.any():
        return None
    return int(lk[mask].iloc[0]["row"])

def _union_candidates_hybrid(cand_from_emb, q_emb_vec, X, q_tfidf_vec, emb, tfidf_top=0):
    # Start with ANN candidates
    cand = np.array(cand_from_emb, dtype=np.int64)

    if X is not None and q_tfidf_vec is not None and tfidf_top and tfidf_top > 0:
        # Full TF-IDF scan to collect top tfidf_top candidates
        scores_full = (X @ q_tfidf_vec.T).toarray().ravel()
        top = min(tfidf_top, scores_full.shape[0])
        tf_top_idx = np.argpartition(-scores_full, range(top))[:top]
        cand = np.unique(np.concatenate([cand, tf_top_idx]))

    # Exact cosine for embeddings (emb is L2 normalized)
    emb_sims = (emb[cand] @ q_emb_vec.ravel()).astype(np.float32)
    # TF-IDF sims for candidates only
    if X is not None and q_tfidf_vec is not None:
        tfidf_sims = (X[cand] @ q_tfidf_vec.T).toarray().ravel().astype(np.float32)
    else:
        tfidf_sims = np.zeros(cand.shape[0], dtype=np.float32)
    return cand, emb_sims, tfidf_sims

def hybrid_search_by_song(out_dir, k=10, candidates=1500, tfidf_top=0, alpha=0.6,
                          title=None, artist=None, model_name="sentence-transformers/all-MiniLM-L6-v2",
                          index_backend="annoy", device="auto"):
    vec, X, emb, meta, lk, index, _ = load_runtime(out_dir, model_name=None, index_backend=index_backend, device=device)
    row = find_song_row(lk, title, artist)
    if row is None:
        print("Song not found. Try adjusting --title/--artist.")
        sys.exit(1)

    # Retrieve candidates via chosen backend
    be = (index_backend or "annoy").lower()
    if be == "hnswlib":
        labels, _ = index.knn_query(emb[row:row+1], k=min(candidates+1, emb.shape[0]))
        cand_emb = labels[0].tolist()
        cand_emb = [i for i in cand_emb if i != row]
    elif be == "annoy":
        cand_emb = index.get_nns_by_item(int(row), n=min(candidates+1, emb.shape[0]), include_distances=False)
        cand_emb = [i for i in cand_emb if i != row]
    elif be == "flat":
        q = emb[row]
        scores = emb @ q
        scores[row] = -np.inf
        top = min(int(candidates), scores.shape[0] - 1)
        cand_emb = np.argpartition(-scores, range(top))[:top].tolist()
    else:
        raise ValueError(f"Unsupported index backend: {index_backend}")

    q_emb_vec = emb[row:row+1]
    q_tfidf_vec = X[row] if X is not None else None

    cand, emb_sims, tfidf_sims = _union_candidates_hybrid(cand_emb, q_emb_vec, X, q_tfidf_vec, emb, tfidf_top=tfidf_top)
    # Exclude self if it slipped in via tfidf union
    cand_mask = cand != row
    cand, emb_sims, tfidf_sims = cand[cand_mask], emb_sims[cand_mask], tfidf_sims[cand_mask]

    hybrid = alpha * emb_sims + (1.0 - alpha) * tfidf_sims
    order = np.argsort(-hybrid)[:k]
    return cand[order], hybrid[order], emb_sims[order], tfidf_sims[order], meta, row

def hybrid_search_by_text(out_dir, query, k=10, candidates=1500, tfidf_top=0, alpha=0.6,
                          model_name="sentence-transformers/all-MiniLM-L6-v2", max_words=220, batch_size=64,
                          index_backend="annoy", device="auto"):
    vec, X, emb, meta, lk, index, model = load_runtime(out_dir, model_name=model_name, index_backend=index_backend, device=device)
    # Embed query (mean over chunks)
    chunks = chunk_words(clean_lyrics(query), max_words=max_words) or [query]
    q_emb = model.encode(chunks, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True).mean(axis=0, keepdims=True)

    # ANN candidates per backend
    be = (index_backend or "annoy").lower()
    if be == "hnswlib":
        labels, _ = index.knn_query(q_emb, k=min(candidates, emb.shape[0]))
        cand_emb = labels[0].tolist()
    elif be == "annoy":
        cand_emb = index.get_nns_by_vector(q_emb.ravel().tolist(), n=min(candidates, emb.shape[0]), include_distances=False)
    elif be == "flat":
        q = q_emb.ravel()
        scores = emb @ q
        top = min(int(candidates), scores.shape[0])
        cand_emb = np.argpartition(-scores, range(top))[:top].tolist()
    else:
        raise ValueError(f"Unsupported index backend: {index_backend}")

    # TF-IDF query (optional)
    q_tfidf = vec.transform([clean_lyrics(query)]) if vec is not None else None

    cand, emb_sims, tfidf_sims = _union_candidates_hybrid(cand_emb, q_emb, X, q_tfidf, emb, tfidf_top=tfidf_top)
    hybrid = alpha * emb_sims + (1.0 - alpha) * tfidf_sims
    order = np.argsort(-hybrid)[:k]
    return cand[order], hybrid[order], emb_sims[order], tfidf_sims[order], meta

def print_results(meta, idxs, hybrid, emb_s, tfidf_s):
    for rnk, (i, h, es, ts) in enumerate(zip(idxs, hybrid, emb_s, tfidf_s), 1):
        print(f"{rnk:2d}. {meta.iloc[i, 1]} — {meta.iloc[i, 2]}  (id={meta.iloc[i,0]}, hybrid={h:.4f}, emb={es:.4f}, tfidf={ts:.4f})")

def main():
    ap = argparse.ArgumentParser(description="Hybrid (Embeddings + TF-IDF) lyrics recommender")
    sub = ap.add_subparsers(dest="cmd", required=True)

    spb = sub.add_parser("build", help="Build TF-IDF, embeddings, and ANN/Flat index")
    spb.add_argument("--csv", required=True)
    spb.add_argument("--out-dir", required=True)
    spb.add_argument("--id-col", default="id")
    spb.add_argument("--title-col", default="title")
    spb.add_argument("--artist-col", default="artist")
    spb.add_argument("--lyrics-col", default="lyrics")
    spb.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    spb.add_argument("--limit-rows", type=int, default=None, help="Limit number of rows to load from CSV (useful for very large files)")
    spb.add_argument("--index-backend", choices=["annoy", "flat", "hnswlib"], default="annoy",
                     help="ANN backend for embeddings. Prefer 'annoy' on Windows.")
    spb.add_argument("--annoy-trees", type=int, default=50, help="Number of trees for Annoy index (higher=better recall, slower build)")
    spb.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Compute device for embeddings")
    spb.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding computation")
    spb.add_argument("--tfidf", choices=["on", "off"], default="on", help="Build TF-IDF artifacts (off reduces CPU RAM)")
    spb.add_argument("--songs-per-batch", type=int, default=1024, help="Number of songs per embedding batch")
    spb.add_argument("--ngram-low", type=int, default=1, help="Min n-gram for TF-IDF")
    spb.add_argument("--ngram-high", type=int, default=1, help="Max n-gram for TF-IDF")
    spb.add_argument("--max-df", type=float, default=0.6, help="Max document frequency for TF-IDF vocabulary")
    spb.add_argument("--min-df", type=int, default=10, help="Min document frequency for TF-IDF vocabulary")
    spb.add_argument("--max-features", type=int, default=100_000, help="Max TF-IDF features")
    spb.add_argument("--max-words", type=int, default=350, help="Max words per chunk for embeddings")

    sps = sub.add_parser("search", help="Search recommendations (hybrid scoring)")
    sps.add_argument("--out-dir", required=True)
    sps.add_argument("--k", type=int, default=10)
    sps.add_argument("--candidates", type=int, default=1500, help="ANN candidates from embeddings")
    sps.add_argument("--tfidf-top", type=int, default=0, help="Optionally union top N by TF-IDF (full scan)")
    sps.add_argument("--alpha", type=float, default=0.6, help="Weight for embedding sim (TF-IDF=1-alpha)")
    sps.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    sps.add_argument("--index-backend", choices=["annoy", "flat", "hnswlib"], default="annoy",
                     help="Backend used for ANN retrieval")
    sps.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Compute device for query embedding")
    sps.add_argument("--batch-size", type=int, default=64, help="Batch size for query embedding (by-text)")
    spsub = sps.add_subparsers(dest="mode", required=True)

    s1 = spsub.add_parser("by-song")
    s1.add_argument("--title", required=True)
    s1.add_argument("--artist", default=None)

    s2 = spsub.add_parser("by-text")
    s2.add_argument("--query", required=True)

    args = ap.parse_args()
    if args.cmd == "build":
        build(args.csv, args.out_dir, args.id_col, args.title_col, args.artist_col, args.lyrics_col,
              model_name=args.model,
              ngram_low=args.ngram_low, ngram_high=args.ngram_high, max_df=args.max_df,
              min_df=args.min_df, max_features=args.max_features,
              batch_size=args.batch_size, max_words=args.max_words, device=args.device, tfidf=args.tfidf,
              songs_per_batch=args.songs_per_batch,
              limit_rows=args.limit_rows, index_backend=args.index_backend,
              annoy_trees=args.annoy_trees)
    elif args.cmd == "search":
        if args.mode == "by-song":
            idxs, h, es, ts, meta, row = hybrid_search_by_song(
                args.out_dir, k=args.k, candidates=args.candidates, tfidf_top=args.tfidf_top,
                alpha=args.alpha, title=args.title, artist=args.artist, model_name=args.model,
                index_backend=args.index_backend, device=args.device
            )
            print(f"\nQuery: {meta.iloc[row,1]} — {meta.iloc[row,2]}")
            print_results(meta, idxs, h, es, ts)
        else:
            idxs, h, es, ts, meta = hybrid_search_by_text(
                args.out_dir, args.query, k=args.k, candidates=args.candidates, tfidf_top=args.tfidf_top,
                alpha=args.alpha, model_name=args.model, index_backend=args.index_backend,
                device=args.device, batch_size=args.batch_size
            )
            print(f"\nQuery text: {args.query[:80]}{'...' if len(args.query)>80 else ''}")
            print_results(meta, idxs, h, es, ts)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
