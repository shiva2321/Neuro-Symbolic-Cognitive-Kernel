# NSCK Knowledge Packs

Pre-built knowledge packs for seeding the NSCK knowledge base.

## ConceptNet Pack

`conceptnet_en_50k.kp` — Top 50K English ConceptNet concepts (schema_version=2, gzip+JSON).

Generate with:
```bash
python nsck/scripts/seed_conceptnet.py path/to/conceptnet_assertions.csv
```

## BERT Pack

`bert_base_uncased.kp` — BERT-base-uncased VSA projections.

Generate with:
```bash
python nsck/scripts/seed_bert.py
```
