# Hackathon Changelog v1.02

This changelog is student-facing only.

Scope:
- `src/map_api.py`
- `src/map_api_core.py`
- hackathon documentation in `docs/`

It does not document organizer-only generation internals beyond behavior that is already reflected in the student-facing hackathon documentation.

## v1.02

### Rules and Timing
- The competition timeout is now documented as `maximumTime = 1000000` seconds.

### Current Robot Profiles
- Changed agents profiles:
    - Drone:
        - `max_velocity = 1.0`
        - `power_draw = 0.02`
        - `battery_recharge = 0.002`
    - Scout:
        - `max_velocity = 0.05`
    - Rover:
        - `max_velocity = 0.01`

### Documentation Updates
- `docs/hackathon_rules.md` now reflects the current timeout and current robot runtime parameters.
- `docs/hackathon_map_api_guide.md` now reflects the current student-facing robot profiles and timing notes.