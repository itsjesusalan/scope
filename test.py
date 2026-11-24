from scope import SCoPEDistances

supports = {0: ["holi", "hola"], 1: ["adios", "adio"]}

query = "ola"

model = SCoPEDistances(
    compressors=["gzip", "bz2"], dissimilarity_metrics=["clm", "ncd"]
)

pred = model.predict(kw_samples=supports, queries=query)[0]

print(pred)
