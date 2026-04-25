# Hackathon Changelog v1.03

This changelog is student-facing only.

Scope:
- `src/map_api.py`
- `src/map_api_core.py`
- hackathon documentation in `docs/`

It does not document organizer-only generation internals beyond behavior that is already reflected in the student-facing hackathon documentation.

## v1.03

### Scoring and Success Criterion
- Updated `docs/hackathon_rules.md` to specify scoring penalty weights:
    - $\lambda_s = 50,000$ (per stuck event)
    - $\lambda_d = 7,000$ (per cell of distance to target)
- Clarified that distance is measured as CELL DISTANCE (number of cells between rover and target).
- Added explicit success criterion: mission is successful if the rover is in the same cell as the target at the end of the run ($D_i^{\text{final}} = 0$).
