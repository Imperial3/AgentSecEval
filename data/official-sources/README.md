# Official Source Collection

AgentSecEval v0.1 uses authoritative public information from official U.S. government sources.

This directory contains normalized frozen source records representing the benchmark-relevant evidence available on the September 2, 2026 snapshot date.

## Included Sources

- `sbir-eligibility-faq.md`
  - SBA/SBIR eligibility requirements

- `sbir-apply.md`
  - SBIR/STTR application eligibility and nonprofit participation rules

- `sbir-policy.md`
  - SBIR/STTR performance-of-work requirements

- `nsf-26-510.md`
  - Current NSF 26-510 solicitation requirements and deadlines

- `charles-river-portfolio.md`
  - Charles River Analytics SBIR/STTR company portfolio record

- `forcefields-award.md`
  - Charles River Analytics FORCEFIELDS award record

- `galois-portfolio.md`
  - Galois SBIR/STTR company portfolio record

- `mapache-award.md`
  - Galois MAPACHE award record

## Source Provenance

Each normalized record includes:

- Source ID
- Publisher
- Authority classification
- Retrieval date
- Canonical source URL
- Benchmark-relevant normalized evidence

The relationship between source IDs, URLs, and local files is recorded in:

`data/source-registry.json`

## Why Local Source Records Are Used

Live webpages can change after an evaluation is conducted.

AgentSecEval therefore maintains normalized local records so that different agent configurations can receive a stable body of evidence during controlled experiments.

These files are not intended to reproduce entire government webpages.

They preserve only the factual material needed for the benchmark tasks while retaining links to the original authoritative records.

## Adversarial Sources

Synthetic attack fixtures are not stored in this directory.

They are kept separately under:

`data/adversarial-sources/`

This separation makes the provenance of official evidence and synthetic adversarial content explicit for benchmark maintainers.
