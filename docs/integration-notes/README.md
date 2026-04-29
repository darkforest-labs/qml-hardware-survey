# Integration notes

One markdown file per backend. Capture: setup steps, gotchas, quirks,
unexpected errors, observed wall-times. This is half the point of the survey.

Suggested template per file:

```
# <backend name>

## Setup
- Plugin / package versions
- AWS / region prerequisites

## Run command
- Exact `qmlsurvey ...` invocation used

## Observations
- Wall time per epoch
- Cost actually billed vs. estimate
- Any errors and how they were resolved
- Anything weird about gradients / shot noise / queue times
```
