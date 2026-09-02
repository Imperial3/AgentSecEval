# Official Source Collection

AgentSecEval v0.1 uses authoritative public information from official U.S. government sources.

Primary sources include:

- SBIR.gov eligibility guidance
- SBA SBIR/STTR policy guidance
- NSF solicitation 26-510
- SBIR.gov company portfolio records
- SBIR.gov award records

The benchmark snapshot date is September 2, 2026.

The source registry in `data/source-registry.json` records each source URL and its relationship to benchmark tasks.

Future versions may store normalized source snapshots to improve reproducibility when live government pages change.

Synthetic adversarial material is stored separately under `data/adversarial-sources/` and is explicitly labeled as synthetic.