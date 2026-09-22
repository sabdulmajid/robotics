# Simplified Technical English Style

This project uses an ASD-STE100 style for its public status documents.

The style makes technical results easier to review.

It does not certify full compliance with the ASD-STE100 specification.

Full compliance also requires a trained review of the controlled dictionary.

## Rules

- Use one term for one meaning.
- Use active voice when possible.
- Put one main idea in each sentence.
- Use no more than 25 words in a descriptive sentence.
- Put a condition before the related action.
- Use digits for measured values.
- Give specific quantities instead of subjective emphasis.
- Use tables for repeated values.
- Identify assumptions, limits, and non-deployable baselines.

OpenPI, LIBERO, SigLIP, MuJoCo, and metric names are approved technical names for this project.

Run this check before a commit:

```bash
python scripts/check_ste_docs.py
```

The check finds mechanical errors.

It does not replace a technical review or a dictionary review.
