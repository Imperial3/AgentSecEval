# AgentSecEval Dataset Methodology

## Objective

AgentSecEval evaluates whether explicit verification mechanisms improve the reliability and security of tool-using language-model agents operating over noisy, incomplete, conflicting, and adversarial information.

## Version

v0.1

## Snapshot Date

September 2, 2026

## Initial Dataset

The first benchmark contains 15 tasks.

Task families include:

1. Eligibility verification
2. Regulatory reasoning
3. Temporal reasoning
4. Procedural reasoning
5. Entity and award extraction
6. Evidence sufficiency and abstention
7. Indirect prompt-injection resistance
8. Source-authority reasoning

## Ground Truth

Gold answers were manually derived from authoritative public sources before evaluation.

Gold answers are stored separately from task inputs.

Agents evaluated against this benchmark must not have access to `gold/answers.json`.

## Source Selection

Official government sources are preferred whenever available.

Current primary sources include SBIR.gov and NSF.gov.

## Adversarial Data

Adversarial sources in the benchmark are synthetic test fixtures.

They are not presented as genuine public webpages.

They are designed to test whether an agent treats instructions contained in retrieved external content as untrusted data rather than higher-priority instructions.

## Limitations

The initial 15-task dataset is intentionally small.

Results from v0.1 should be treated as preliminary rather than as evidence of broad agent performance.

Future versions will expand the dataset and diversify task domains.